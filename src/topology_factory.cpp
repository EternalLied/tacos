/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include "topology_factory.h"
#include <tacos/topology/mesh_2d.h>
#include <tacos/topology/switch_clique.h>
#include <tacos/topology/dgx_topologies.h>
#include <tacos/topology/ndv2_topologies.h>
#include <tacos/topology/amd_topologies.h>
#include <map>
#include <functional>
#include <sstream>

namespace tacos {

std::optional<Topology> createTopology(const std::string& topologyName) {
    // Define topology factory functions
    static const std::map<std::string, std::function<Topology()>> topologyFactories = {
        // Mesh 2D
        {"mesh2d", []() {
            const auto width = 4;
            const auto height = 3;
            const auto latency = 0.5;  // microseconds (us)
            const auto bandwidth = 50;  // GiB/sec
            return Mesh2D(width, height, bandwidth, latency);
        }},
        
        // Switch Clique
        {"switch_clique", []() {
            Topology topo;
            auto temp = SwitchClique(
                /*npus*/8,
                /*BW*/50.0,
                /*α_g2s*/0.7,
                /*α_s2g*/0.7,
                /*allowCopy*/false,
                /*maxParallel*/-1,
                /*forwardingMode*/SwitchForwardingMode::CUT_THROUGH
            );
            topo = temp;
            return topo;
        }},
        
        // DGX1 TE-CCL (8 GPUs, direct GPU-GPU graph)
        {"DGX1_Tecc", []() {
            Topology topo;
            BuildDGX1_Tecc(topo, /*alpha_us*/0.7);
            return topo;
        }},
        
        // DGX1 Single Chassis
        {"DGX1_1", []() {
            Topology topo;
            BuildDGX1_SingleChassis(topo, /*bw_gbps*/125.0, /*alpha_us*/0.35, /*allow_copy*/false);
            return topo;
        }},
        
        // DGX2 Two Chassis TE-CCL
        {"DGX2_2_Tecc", []() {
            Topology topo;
            BuildDGX2_TwoChassis_Tecc(topo, /*allow_copy*/false);
            return topo;
        }},
        
        // DGX2 Two Chassis
        {"DGX2_2", []() {
            Topology topo;
            BuildDGX2_TwoChassis(topo, /*allow_copy*/false);
            return topo;
        }},
        
        // NDv2 Two Chassis TE-CCL
        {"NDv2_2_Tecc", []() {
            Topology topo;
            BuildNDv2_TwoChassis_Tecc(topo);
            return topo;
        }},
        
        // NDv2 Two Chassis
        {"NDv2_2", []() {
            Topology topo;
            BuildNDv2_TwoChassis(topo, /*allow_copy*/false);
            return topo;
        }},
        
        // NDv2 Four Chassis TE-CCL
        {"NDv2_4_Tecc", []() {
            Topology topo;
            BuildNDv2_FourChassis_Tecc(topo, /*allow_copy*/false);
            return topo;
        }},
        
        // NDv2 Four Chassis
        {"NDv2_4", []() {
            Topology topo;
            BuildNDv2_FourChassis(topo, /*allow_copy*/false);
            return topo;
        }},
        
        // AMD MI250 Two Chassis
        {"AMD_2", []() {
            Topology topo;
            BuildAMD_MI250_2Chassis(topo, /*allow_copy*/false);
            return topo;
        }},
        
        // AMD MI250 Four Chassis
        {"AMD_4", []() {
            Topology topo;
            BuildAMD_MI250_4Chassis(topo, /*allow_copy*/false);
            return topo;
        }}
    };
    
    auto it = topologyFactories.find(topologyName);
    if (it == topologyFactories.end()) {
        return std::nullopt;
    }
    
    auto topology = it->second();
    
    // Call finalizeReachability_() to compute GPU-to-GPU reachability and backtrack maps
    // This is essential and must be called before using the topology for scheduling
    topology.finalizeReachability_();
    
    return topology;
}

std::string getAvailableTopologies() {
    static const std::vector<std::string> topologies = {
        "mesh2d", "switch_clique", 
        "DGX1_Tecc", "DGX1_1", 
        "DGX2_2_Tecc", "DGX2_2", 
        "NDv2_2_Tecc", "NDv2_2", 
        "NDv2_4_Tecc", "NDv2_4", 
        "AMD_2", "AMD_4"
    };
    
    std::ostringstream oss;
    for (size_t i = 0; i < topologies.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << topologies[i];
    }
    return oss.str();
}

}  // namespace tacos
