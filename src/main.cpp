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

int main(int argc, char* argv[]) {
    // set print precision
    fixed(std::cout);
    std::cout.precision(2);

    // Parse command line arguments for topology, collective, and multi-round synthesis
    std::string topologyName = "switch_clique";  // default topology
    std::string collectiveName = "allgather";  // default collective
    bool enableMultiRound = false;
    int maxNoImprovementRounds = 10;  // default max rounds without improvement
    
    if (argc > 1) {
        topologyName = argv[1];
    }
    if (argc > 2) {
        collectiveName = argv[2];
    }
    if (argc > 3) {
        std::string multiRoundArg = argv[3];
        if (multiRoundArg == "multi" || multiRoundArg == "enable") {
            enableMultiRound = true;
        }
    }
    if (argc > 4) {
        try {
            maxNoImprovementRounds = std::stoi(argv[4]);
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
    const Collective::ChunkSize outputBufferSize = 12 * (1 << 20);  // 12 MiB
    const auto collectivesCount = 1;  // number of rounds (can be scheduled concurrently)
    
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
        // For AllToAll, each NPU sends N chunks (one to each NPU)
        // Total data per NPU = outputBufferSize, divided into N chunks
        chunkSize = outputBufferSize / npusCount;
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
        
        // Calculate algorithm bandwidth (GB/s)
        // Bandwidth = Total data size (bytes) / Time (us) * 1e6 (us to s) / 1e9 (bytes to GB)
        //           = Total data size / Time / 1000
        double algoBandwidth = (result.bestCollectiveTime > 0.0) ? 
            (static_cast<double>(totalDataSize) / result.bestCollectiveTime / 1000.0) : 0.0;
        
        std::cout << "=== Multi-Round Synthesis Summary ===" << std::endl;
        std::cout << "Completed rounds: " << result.totalRounds << std::endl;
        std::cout << "Average synthesis time: " << result.totalSynthesisTime / 1000 / result.totalRounds << " ms" << std::endl;
        std::cout << "Best collective time: " << result.bestCollectiveTime 
                  << " us (found at round " << result.bestRound << ")" << std::endl;
        std::cout << "Algorithm bandwidth: " << std::fixed << std::setprecision(2) 
                  << algoBandwidth << " GB/s" << std::endl;
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
        
        // Calculate algorithm bandwidth (GB/s)
        // Bandwidth = Total data size (bytes) / Time (us) * 1e6 (us to s) / 1e9 (bytes to GB)
        //           = Total data size / Time / 1000
        double algoBandwidth = (collectiveTime > 0.0) ? 
            (static_cast<double>(totalDataSize) / collectiveTime / 1000.0) : 0.0;
        
        std::cout << std::endl;
        std::cout << "Time to solve: " << time / 1000 << " ms" << std::endl;
        std::cout << "Collective Time: " << collectiveTime << " us" << std::endl;
        std::cout << "Algorithm bandwidth: " << std::fixed << std::setprecision(2) 
                  << algoBandwidth << " GB/s" << std::endl;
        std::cout << "Average link utilization: " << std::fixed << std::setprecision(2) 
                  << linkUtilization << "%" << std::endl;
    }

    // terminate
    return 0;
}
