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

    // const auto topology = Mesh2D(width, height, bandwidth, latency);
    const auto topology = SwitchClique(/*npus*/8, /*BW*/50.0, /*α_g2s*/0.7, /*α_s2g*/0.7);

    const auto npusCount = topology.npusCount();
    std::cout << "NPUs count: " << npusCount << std::endl;
    
    // print link statistics
    const auto logicalLinks = topology.linksCount();
    const auto physicalLinks = topology.physicalLinksCount();
    const auto switchCount = topology.switchesCount();
    std::cout << "Logical links (GPU-to-GPU): " << logicalLinks << std::endl;
    std::cout << "Physical links (actual): " << physicalLinks << std::endl;
    std::cout << "Switches count: " << switchCount << std::endl;

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
