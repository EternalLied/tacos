/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include "theoretical_bound.h"
#include <limits>
#include <algorithm>
#include <map>
#include <vector>

namespace tacos {

double calculateTheoreticalLowerBound(const Topology& topology, 
                                     const Collective& collective,
                                     Collective::ChunkSize chunkSize) {
    // Create TEN for accurate hop count calculation
    auto ten = TimeExpandedNetwork(topology, chunkSize);
    
    const int npusCount = topology.npusCount();
    const int totalLinks = topology.physLinksCount();
    const int chunksCount = collective.chunksCount();
    
    const auto& physLinks = topology.physLinks();
    
    // Strategy: Find the bottleneck bandwidth in the topology
    // For multi-level topologies, the bottleneck is the minimum bandwidth link(s)
    // that all cross-traffic must pass through
    
    double totalLatency = 0.0;
    int linkCount = 0;
    
    // Group links by bandwidth to identify bottleneck tier
    std::map<double, std::vector<int>> bandwidthGroups;  // bandwidth -> list of link indices
    
    for (size_t i = 0; i < physLinks.size(); ++i) {
        const auto& link = physLinks[i];
        bandwidthGroups[link.bw].push_back(i);
        totalLatency += link.alpha;
        linkCount++;
    }
    
    // Find minimum bandwidth (bottleneck tier)
    double minBandwidth = std::numeric_limits<double>::max();
    for (const auto& [bw, indices] : bandwidthGroups) {
        if (bw < minBandwidth) {
            minBandwidth = bw;
        }
    }
    
    // Calculate effective bandwidth based on bottleneck analysis
    double totalBandwidth = 0.0;
    
    // Categorize links by type
    std::vector<int> deviceToDeviceLinks;
    std::vector<int> deviceToSwitchLinks;
    std::vector<int> switchToDeviceLinks;
    std::vector<int> switchToSwitchLinks;
    
    for (size_t i = 0; i < physLinks.size(); ++i) {
        const auto& link = physLinks[i];
        const bool srcIsDevice = (link.src < npusCount);
        const bool dstIsDevice = (link.dst < npusCount);
        
        if (srcIsDevice && dstIsDevice) {
            deviceToDeviceLinks.push_back(i);
        } else if (srcIsDevice && !dstIsDevice) {
            deviceToSwitchLinks.push_back(i);
        } else if (!srcIsDevice && dstIsDevice) {
            switchToDeviceLinks.push_back(i);
        } else {
            switchToSwitchLinks.push_back(i);
        }
    }
    
    // Determine effective bandwidth:
    // 1. If there are switch-to-switch links, they are likely the bottleneck
    // 2. Otherwise, use device-to-switch parallelism
    
    if (!switchToSwitchLinks.empty()) {
        // Multi-level topology: switch-to-switch links are typically the bottleneck
        // Sum up all switch-to-switch bandwidth (bidirectional considered)
        double switchToSwitchBandwidth = 0.0;
        for (int idx : switchToSwitchLinks) {
            switchToSwitchBandwidth += physLinks[idx].bw;
        }
        totalBandwidth = switchToSwitchBandwidth;
        
    } else if (!deviceToSwitchLinks.empty() && !switchToDeviceLinks.empty()) {
        // Single-level switched topology: bottleneck is min(uplink, downlink)
        double uplinkBandwidth = 0.0;
        double downlinkBandwidth = 0.0;
        
        for (int idx : deviceToSwitchLinks) {
            uplinkBandwidth += physLinks[idx].bw;
        }
        for (int idx : switchToDeviceLinks) {
            downlinkBandwidth += physLinks[idx].bw;
        }
        
        totalBandwidth = std::min(uplinkBandwidth, downlinkBandwidth);
        
    } else if (!deviceToDeviceLinks.empty()) {
        // Direct device-to-device topology (mesh/torus, no switches)
        // Bottleneck is node degree, not total link bandwidth
        
        // Calculate per-node bandwidth (outgoing links per device)
        std::vector<double> nodeOutBandwidth(npusCount, 0.0);
        std::vector<double> nodeInBandwidth(npusCount, 0.0);
        
        for (int idx : deviceToDeviceLinks) {
            const auto& link = physLinks[idx];
            nodeOutBandwidth[link.src] += link.bw;
            nodeInBandwidth[link.dst] += link.bw;
        }
        
        // For AllToAll, each node needs to send and receive data
        // The bottleneck is the minimum of (sum of all node bandwidths / 2)
        // Divide by 2 because each node must both send and receive
        double totalOutBandwidth = 0.0;
        double totalInBandwidth = 0.0;
        
        for (int i = 0; i < npusCount; ++i) {
            totalOutBandwidth += nodeOutBandwidth[i];
            totalInBandwidth += nodeInBandwidth[i];
        }
        
        // Effective bandwidth is limited by the direction with less aggregate bandwidth
        // Further divided by 2 since each node must share time between send/receive
        totalBandwidth = std::min(totalOutBandwidth, totalInBandwidth) / 2.0;
    }
    
    // Ensure bandwidth is positive
    if (totalBandwidth <= 0) {
        totalBandwidth = 1.0;  // Fallback to avoid division by zero
    }
    
    // Calculate average latency (arithmetic mean)
    double avgLatency = (linkCount > 0) ? (totalLatency / linkCount) : 0.0;
    
    // Calculate effective parallel link count for latency calculation
    // Use the categorized links from above
    int effectiveParallelLinks = 0;
    
    if (!switchToSwitchLinks.empty()) {
        // Multi-level topology: bottleneck is switch-to-switch links
        effectiveParallelLinks = switchToSwitchLinks.size();
    } else {
        // Single-level or direct topology
        effectiveParallelLinks = deviceToDeviceLinks.size() + 
                                std::min(deviceToSwitchLinks.size(), switchToDeviceLinks.size());
    }
    
    // Ensure at least 1 to avoid division by zero
    if (effectiveParallelLinks == 0) {
        effectiveParallelLinks = 1;
    }
    
    // Detect topology partitions (local domains) by analyzing device-to-switch connectivity
    // Devices connected to the same switch are in the same partition
    std::vector<int> devicePartition(npusCount, -1);  // device -> partition ID
    
    if (!switchToSwitchLinks.empty()) {
        // Multi-level topology: partition by first-hop switch
        int partitionId = 0;
        std::map<int, int> switchToPartition;  // switch ID -> partition ID
        
        for (int dev = 0; dev < npusCount; ++dev) {
            // Find which switch this device connects to
            for (int idx : deviceToSwitchLinks) {
                const auto& link = physLinks[idx];
                if (link.src == dev) {
                    int switchNode = link.dst;
                    
                    // Assign partition based on switch
                    if (switchToPartition.find(switchNode) == switchToPartition.end()) {
                        switchToPartition[switchNode] = partitionId++;
                    }
                    devicePartition[dev] = switchToPartition[switchNode];
                    break;
                }
            }
        }
    } else {
        // Single-level topology: all devices in same partition
        for (int dev = 0; dev < npusCount; ++dev) {
            devicePartition[dev] = 0;
        }
    }
    
    // Count unsatisfied postconditions and calculate LOGICAL data volume
    // For AllToAll (no data forwarding), each transfer occupies links equal to its logical hop count
    int totalTransfers = 0;
    int crossPartitionTransfers = 0;
    double totalLogicalHops = 0.0;  // Weighted by logical hop count
    double crossPartitionLogicalHops = 0.0;
    
    CollectiveType collectiveType = collective.getType();
    
    for (int chunk = 0; chunk < chunksCount; ++chunk) {
        const auto& dests = collective.postcondition(chunk);
        const auto src = collective.precondition(chunk);
        
        for (const auto dest : dests) {
            if (src != dest) {
                totalTransfers++;
                
                // Get logical hop count for this transfer
                int logicalHops = ten.getRouteHops(src, dest);
                if (logicalHops <= 0) logicalHops = 1;  // At least 1 hop
                
                // For AllToAll: each transfer uses 'logicalHops' links
                // For AllGather: data can be forwarded, so only count 1
                double hopWeight = (collectiveType == CollectiveType::ALL_TO_ALL) ? logicalHops : 1.0;
                
                totalLogicalHops += hopWeight;
                
                // Check if this is a cross-partition transfer
                if (devicePartition[src] != devicePartition[dest]) {
                    crossPartitionTransfers++;
                    crossPartitionLogicalHops += hopWeight;
                }
            }
        }
    }
    
    // Calculate LOGICAL data volumes (accounting for multi-hop paths)
    double totalDataBytes = totalLogicalHops * chunkSize;
    double crossPartitionDataBytes = crossPartitionLogicalHops * chunkSize;
    
    double totalDataGiB = totalDataBytes / (1024.0 * 1024.0 * 1024.0);
    double crossPartitionDataGiB = crossPartitionDataBytes / (1024.0 * 1024.0 * 1024.0);
    
    // Calculate transmission time
    double transmissionTime = 0.0;
    
    if (!switchToSwitchLinks.empty() && crossPartitionTransfers > 0) {
        // Multi-level: bottleneck is cross-partition bandwidth
        // Only cross-partition data is limited by switch-to-switch bandwidth
        transmissionTime = crossPartitionDataGiB / totalBandwidth;  // in seconds
    } else {
        // Single-level or no cross-partition traffic: use total bandwidth
        transmissionTime = totalDataGiB / totalBandwidth;  // in seconds
    }
    
    transmissionTime *= 1e6;  // convert to microseconds
    
    // Calculate static latency component based on collective type
    // For AllToAll with multi-hop paths, latency is already accounted for in physical hop weighting
    // Use a simplified latency model: average hops × average latency × serialization factor
    double staticLatency = 0.0;
    
    if (collectiveType == CollectiveType::ALL_GATHER) {
        // AllGather: static latency based on cross-partition transfers if multi-level
        int relevantTransfers = (!switchToSwitchLinks.empty()) ? crossPartitionTransfers : totalTransfers;
        staticLatency = (static_cast<double>(relevantTransfers) / effectiveParallelLinks) * avgLatency;
        
    } else if (collectiveType == CollectiveType::ALL_TO_ALL) {
        // AllToAll: latency is proportional to logical hops (each hop adds latency)
        // Average logical hops per transfer
        double avgLogicalHops = (totalTransfers > 0) ? (totalLogicalHops / totalTransfers) : 1.0;
        
        // Static latency: (number of serialized transfers) × (avg hops) × (per-hop latency)
        // Serialization factor depends on whether multi-level or not
        int relevantTransfers = (!switchToSwitchLinks.empty()) ? crossPartitionTransfers : totalTransfers;
        staticLatency = (static_cast<double>(relevantTransfers) / effectiveParallelLinks) * avgLogicalHops * avgLatency;
    }
    
    return transmissionTime + staticLatency;
}

}  // namespace tacos
