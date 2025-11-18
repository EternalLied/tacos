/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <iostream>
#include <tacos/collective/all_gather.h>
#include <tacos/event_queue/timer.h>
#include <tacos/synthesizer/synthesizer.h>
#include <tacos/topology/mesh_2d.h>
#include <tacos/topology/switch_clique.h>
#include <tacos/topology/multichassis_presets.h>
#include "log.h"

using namespace tacos;

int main() {
    // set print precision
    fixed(std::cout);
    std::cout.precision(2);

    // construct a topology
    const auto width = 4;
    const auto height = 3;

    const auto latency = 0.5;  // microseconds (us)
    const auto bandwidth = 50;  // GiB/sec

    const auto mesh2d = Mesh2D(width, height, bandwidth, latency);
    const auto switch_clique = SwitchClique(/*npus*/8, /*BW*/50.0, /*α_g2s*/0.7, /*α_s2g*/0.7);

    tacos::Topology DGX1_1;
    tacos::BuildDGX1_SingleChassis(DGX1_1, /*bw_gbps*/125.0, /*alpha_us*/0.35, /*allow_copy*/false);

    tacos::Topology DGX2_2;
    tacos::BuildDGX2_TwoChassis(DGX2_2, /*allow_copy*/false);

    tacos::Topology NDv2_2;
    tacos::BuildNDv2_FourChassis(NDv2_2, /*allow_copy*/false);

    tacos::Topology NDv2_4;
    tacos::BuildNDv2_FourChassis(NDv2_4, /*allow_copy*/false);

    tacos::Topology AMD_2;
    tacos::BuildAMD_MI250_2Chassis(AMD_2, /*allow_copy*/false);

    tacos::Topology AMD_4;
    tacos::BuildAMD_MI250_4Chassis(AMD_4, /*allow_copy*/false);

    const auto topology = switch_clique;
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
    const auto collectivesCount = 1;  // initial chunks per each NPU

    const auto collective = AllGather(npusCount, collectivesCount);
    const auto chunkSize = outputBufferSize / (npusCount * collectivesCount);
    const auto chunksCount = collective.chunksCount();
    std::cout << "Chunks count: " << chunksCount << std::endl;
    std::cout << "Each chunk size: " << chunkSize << " bytes" << std::endl;
    DebugLog(std::cout << std::endl);

    // create timer
    auto synthesizerTimer = Timer();

    // create synthesizer and solve
    synthesizerTimer.start();
    auto synthesizer = Synthesizer();
    auto collectiveTime = synthesizer.solve(topology, collective, chunkSize);
    synthesizerTimer.stop();

    // print result
    auto time = synthesizerTimer.time();
    std::cout << std::endl;
    std::cout << "Time to solve: " << time / 1000 << " ms" << std::endl;
    std::cout << "Collective Time: " << collectiveTime << " us" << std::endl;

    // terminate
    return 0;
}
