/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <limits>
#include <csignal>
#include <atomic>
#include <tacos/collective/all_gather.h>
#include <tacos/collective/all_to_all.h>
#include <tacos/event_queue/timer.h>
#include <tacos/synthesizer/synthesizer.h>
#include <tacos/synthesizer/time_expanded_network.h>
#include "log.h"
#include "theoretical_bound.h"
#include "topology_factory.h"

using namespace tacos;

// Global variable for signal handling
std::atomic<bool> g_interrupted(false);

void signalHandler(int signum) {
    g_interrupted.store(true);
    std::cout << "\n\n=== Interrupted by user (Ctrl+C) ===" << std::endl;
    std::cout << "Stopping synthesis gracefully..." << std::endl;
    // Don't exit immediately, let the solver finish current round and return result
}

// Parse data size string with units (e.g., "12MB", "1GB", "512KB")
// Returns size in bytes, or -1 on error
int64_t parseDataSize(const std::string& sizeStr) {
    if (sizeStr.empty()) return -1;
    
    // Find where the numeric part ends
    size_t numEnd = 0;
    while (numEnd < sizeStr.length() && 
           (std::isdigit(sizeStr[numEnd]) || sizeStr[numEnd] == '.')) {
        ++numEnd;
    }
    
    if (numEnd == 0) return -1;  // No number found
    
    // Parse the numeric part
    double value = 0.0;
    try {
        value = std::stod(sizeStr.substr(0, numEnd));
    } catch (...) {
        return -1;
    }
    
    if (value < 0) return -1;
    
    // Parse the unit part (case-insensitive)
    std::string unit = sizeStr.substr(numEnd);
    // Convert to uppercase for comparison
    for (char& c : unit) {
        c = std::toupper(c);
    }
    
    // Remove 'B' suffix if present (e.g., "MB" -> "M")
    if (!unit.empty() && unit.back() == 'B') {
        unit.pop_back();
    }
    
    // Apply multiplier based on unit
    int64_t multiplier = 1;
    if (unit.empty() || unit == "B") {
        multiplier = 1;  // bytes
    } else if (unit == "K" || unit == "KB") {
        multiplier = 1 << 10;  // 1024 bytes
    } else if (unit == "M" || unit == "MB") {
        multiplier = 1 << 20;  // 1048576 bytes
    } else if (unit == "G" || unit == "GB") {
        multiplier = 1 << 30;  // 1073741824 bytes
    } else {
        return -1;  // Unknown unit
    }
    
    return static_cast<int64_t>(value * multiplier);
}

int main(int argc, char* argv[]) {
    // set print precision
    fixed(std::cout);
    std::cout.precision(2);

    // Parse command line arguments
    // Usage: tacos <topology> <collective> [data_size] [multi] [max_rounds]
    // Example: tacos DGX2_2 allgather 12MB multi 100
    std::string topologyName = "switch_clique";  // default topology
    std::string collectiveName = "allgather";  // default collective
    bool enableMultiRound = false;
    int maxNoImprovementRounds = 10;  // default max rounds without improvement
    int64_t dataSizeBytes = 1 * (1 << 20);  // default: 1 MiB
    
    if (argc > 1) {
        topologyName = argv[1];
    }
    if (argc > 2) {
        collectiveName = argv[2];
    }
    if (argc > 3) {
        // Try to parse as data size first
        int64_t parsedSize = parseDataSize(argv[3]);
        if (parsedSize > 0) {
            dataSizeBytes = parsedSize;
        } else {
            // If not a valid data size, check if it's "multi"
            std::string arg3 = argv[3];
            if (arg3 == "multi" || arg3 == "enable") {
                enableMultiRound = true;
            } else {
                std::cerr << "Warning: Unrecognized argument '" << argv[3] 
                          << "', expected data size (e.g., 12MB) or 'multi'" << std::endl;
            }
        }
    }
    if (argc > 4) {
        std::string arg4 = argv[4];
        if (arg4 == "multi" || arg4 == "enable") {
            enableMultiRound = true;
        } else {
            try {
                maxNoImprovementRounds = std::stoi(argv[4]);
                if (maxNoImprovementRounds < 1) {
                    std::cerr << "Warning: Invalid max no-improvement rounds, using default (10)" << std::endl;
                    maxNoImprovementRounds = 10;
                }
            } catch (...) {
                std::cerr << "Warning: Invalid argument format '" << argv[4] << "'" << std::endl;
            }
        }
    }
    if (argc > 5) {
        try {
            maxNoImprovementRounds = std::stoi(argv[5]);
            if (maxNoImprovementRounds < 1) {
                std::cerr << "Warning: Invalid max no-improvement rounds, using default (10)" << std::endl;
                maxNoImprovementRounds = 10;
            }
        } catch (...) {
            std::cerr << "Warning: Invalid max no-improvement rounds format, using default (10)" << std::endl;
        }
    }

    // Create selected topology
    std::cout << "Selected topology: " << topologyName << std::endl;
    auto topologyOpt = createTopology(topologyName);
    if (!topologyOpt) {
        std::cerr << "Error: Unknown topology '" << topologyName << "'" << std::endl;
        std::cerr << "Available topologies: " << getAvailableTopologies() << std::endl;
        return 1;
    }
    auto topology = std::move(*topologyOpt);
    
    const auto npusCount = topology.npusCount();
    std::cout << "NPUs count: " << npusCount << std::endl;
    
    // print switch count
    const auto switchCount = topology.switchesCount();
    std::cout << "Switches count: " << switchCount << std::endl;

    // print link statistics
    const auto totalLinks = topology.physLinksCount();
    std::cout << "Total physical links: " << totalLinks << std::endl;
    
    const auto [d2d, d2s, s2d, s2s] = topology.getLinkStatistics();
    std::cout << "  Device-to-Device links: " << d2d << std::endl;
    std::cout << "  Device-to-Switch links: " << d2s << std::endl;
    std::cout << "  Switch-to-Device links: " << s2d << std::endl;
    std::cout << "  Switch-to-Switch links: " << s2s << std::endl;

    // create collective
    const Collective::ChunkSize outputBufferSize = dataSizeBytes;  // from command line or default
    const auto collectivesCount = 1;  // number of rounds (can be scheduled concurrently)
    
    std::cout << "Data size: " << outputBufferSize / (1 << 20) << " MiB (" 
              << outputBufferSize << " bytes)" << std::endl;
    
    std::unique_ptr<Collective> collective;
    Collective::ChunkSize chunkSize;
    Collective::ChunkSize totalDataSize;  // Total data size for bandwidth calculation
    
    if (collectiveName == "allgather") {
        std::cout << "Selected collective: AllGather" << std::endl;
        collective = std::make_unique<AllGather>(npusCount, collectivesCount);
        chunkSize = outputBufferSize / (npusCount * collectivesCount);
        // For AllGather, total data size is the output buffer size per NPU
        totalDataSize = outputBufferSize;
    } else if (collectiveName == "alltoall") {
        std::cout << "Selected collective: AllToAll" << std::endl;
        collective = std::make_unique<AllToAll>(npusCount, collectivesCount);
        chunkSize = outputBufferSize / (npusCount * collectivesCount);
        // For AllToAll, total data size is the output buffer size per NPU
        totalDataSize = outputBufferSize;
    } else {
        std::cerr << "Error: Unknown collective '" << collectiveName << "'" << std::endl;
        std::cerr << "Available collectives: allgather, alltoall" << std::endl;
        return 1;
    }
    
    const auto chunksCount = collective->chunksCount();
    std::cout << "Chunks count: " << chunksCount << std::endl;
    std::cout << "Each chunk size: " << chunkSize << " bytes" << std::endl;
    std::cout << "Total data size: " << totalDataSize / (1 << 20) << " MiB" << std::endl;
    
    DebugLog(std::cout << std::endl);

    // TODO: Theoretical lower bound calculation is not accurate, commented out for now
    // double theoreticalLowerBound = calculateTheoreticalLowerBound(topology, *collective, chunkSize);
    // std::cout << "Theoretical lower bound: " << theoreticalLowerBound << " us" << std::endl;
    std::cout << std::endl;

    if (enableMultiRound) {
        // Register signal handler for Ctrl+C
        std::signal(SIGINT, signalHandler);
        
        std::cout << "\n=== Multi-Round Synthesis Mode ===" << std::endl;
        std::cout << "Stop after " << maxNoImprovementRounds << " consecutive rounds without improvement" << std::endl;
        std::cout << "(Press Ctrl+C to interrupt and show best result)" << std::endl;
        std::cout << std::endl;
        
        // Run multi-round synthesis
        auto result = Synthesizer::solveMultiRound(topology, *collective, chunkSize, 
                                                    maxNoImprovementRounds, g_interrupted);
        
        // Calculate link utilization for best synthesizer
        double linkUtilization = 0.0;
        if (result.bestSynthesizer) {
            linkUtilization = result.bestSynthesizer->calculateLinkUtilization(topology);
        }
        
        // Calculate algorithm bandwidth in GiB/s (binary)
        // GiB/s = (bytes / time_us) / (2^30 / 1e6) = bytes / time_us / 1073.741824
        const double kGiBPerSecDenom = (1024.0 * 1024.0 * 1024.0) / 1'000'000.0; // 1073.741824
        double algoBandwidth = (result.bestCollectiveTime > 0.0) ?
            (static_cast<double>(totalDataSize) / result.bestCollectiveTime / kGiBPerSecDenom) : 0.0;
        
        std::cout << "=== Multi-Round Synthesis Summary ===" << std::endl;
        std::cout << "Completed rounds: " << result.totalRounds << std::endl;
        std::cout << "Average synthesis time: " << result.totalSynthesisTime / 1000 / result.totalRounds << " ms" << std::endl;
        std::cout << "Best collective time: " << result.bestCollectiveTime 
                  << " us (found at round " << result.bestRound << ")" << std::endl;
    std::cout << "Algorithm bandwidth: " << std::fixed << std::setprecision(2)
          << algoBandwidth << " GiB/s" << std::endl;
        std::cout << "Average link utilization: " << std::fixed << std::setprecision(2) 
                  << linkUtilization << "%" << std::endl;
        
        if (result.interrupted) {
            std::cout << "(Interrupted by user)" << std::endl;
        }
        
    } else {
        // Single-round synthesis (original behavior)
        auto synthesizerTimer = Timer();
        synthesizerTimer.start();
        
        auto synthesizer = Synthesizer();
        auto collectiveTime = synthesizer.solve(topology, *collective, chunkSize);
        
        synthesizerTimer.stop();
        auto time = synthesizerTimer.time();
        
        // Calculate link utilization
        auto linkUtilization = synthesizer.calculateLinkUtilization(topology);
        
        // Calculate algorithm bandwidth in GiB/s (binary)
        const double kGiBPerSecDenom = (1024.0 * 1024.0 * 1024.0) / 1'000'000.0; // 1073.741824
        double algoBandwidth = (collectiveTime > 0.0) ?
            (static_cast<double>(totalDataSize) / collectiveTime / kGiBPerSecDenom) : 0.0;
        
        std::cout << std::endl;
        std::cout << "Time to solve: " << time / 1000 << " ms" << std::endl;
        std::cout << "Collective Time: " << collectiveTime << " us" << std::endl;
    std::cout << "Algorithm bandwidth: " << std::fixed << std::setprecision(2)
          << algoBandwidth << " GiB/s" << std::endl;
        std::cout << "Average link utilization: " << std::fixed << std::setprecision(2) 
                  << linkUtilization << "%" << std::endl;
    }

    // terminate
    return 0;
}
