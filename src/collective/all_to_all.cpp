/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <cassert>
#include <tacos/collective/all_to_all.h>

using namespace tacos;

AllToAll::AllToAll(const int npusCount, const int collectivesCount) noexcept : Collective() {
    assert(npusCount > 0);
    assert(collectivesCount > 0);

    // In All-to-All collective:
    // - Each NPU i has N chunks (where N = npusCount)
    // - Chunk j from NPU i needs to be sent to NPU j (including itself)
    // - Similar to AllGather, postcondition includes source node itself
    // - markPrecondition_() will mark local chunks as already satisfied
    //
    // Total chunks = npusCount * npusCount * collectivesCount
    // For example, with 4 NPUs and 1 round:
    //   NPU 0: chunks [0,1,2,3] -> destinations [0,1,2,3] (chunk 0 local, already satisfied)
    //   NPU 1: chunks [4,5,6,7] -> destinations [0,1,2,3] (chunk 5 local, already satisfied)
    //   NPU 2: chunks [8,9,10,11] -> destinations [0,1,2,3] (chunk 10 local, already satisfied)
    //   NPU 3: chunks [12,13,14,15] -> destinations [0,1,2,3] (chunk 15 local, already satisfied)

    for (int round = 0; round < collectivesCount; ++round) {
        for (int src = 0; src < npusCount; ++src) {
            for (int dest = 0; dest < npusCount; ++dest) {
                // Each chunk goes to exactly one destination (including src itself)
                // Local chunks (src == dest) will be filtered by markPrecondition_()
                auto dests = std::unordered_set<NpuID>();
                dests.insert(dest);
                chunk_(src, dests);
            }
        }
    }
}
