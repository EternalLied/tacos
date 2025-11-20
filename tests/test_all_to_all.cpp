/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <gtest/gtest.h>
#include <tacos/collective/all_to_all.h>

using namespace tacos;

TEST(AllToAllTest, BasicFunctionality) {
    const int npusCount = 4;
    const int collectivesCount = 1;
    
    AllToAll alltoall(npusCount, collectivesCount);
    
    // Total chunks should be npusCount * npusCount * collectivesCount
    EXPECT_EQ(alltoall.chunksCount(), npusCount * npusCount * collectivesCount);
    EXPECT_EQ(alltoall.chunksCount(), 16);
}

TEST(AllToAllTest, PreconditionsAndPostconditions) {
    const int npusCount = 4;
    const int collectivesCount = 1;
    
    AllToAll alltoall(npusCount, collectivesCount);
    
    // Verify that each chunk has correct source and destination
    int chunkId = 0;
    for (int src = 0; src < npusCount; ++src) {
        for (int dest = 0; dest < npusCount; ++dest) {
            // Check precondition (source)
            EXPECT_EQ(alltoall.precondition(chunkId), src);
            
            // Check postcondition (destination set)
            const auto& dests = alltoall.postcondition(chunkId);
            EXPECT_EQ(dests.size(), 1);  // Each chunk goes to exactly one destination
            EXPECT_TRUE(dests.find(dest) != dests.end());
            
            ++chunkId;
        }
    }
}

TEST(AllToAllTest, MultipleRounds) {
    const int npusCount = 3;
    const int collectivesCount = 2;
    
    AllToAll alltoall(npusCount, collectivesCount);
    
    // Total chunks = 3 * 3 * 2 = 18
    EXPECT_EQ(alltoall.chunksCount(), 18);
    
    // Verify first round (chunks 0-8)
    int chunkId = 0;
    for (int round = 0; round < collectivesCount; ++round) {
        for (int src = 0; src < npusCount; ++src) {
            for (int dest = 0; dest < npusCount; ++dest) {
                EXPECT_EQ(alltoall.precondition(chunkId), src);
                const auto& dests = alltoall.postcondition(chunkId);
                EXPECT_EQ(dests.size(), 1);
                EXPECT_TRUE(dests.find(dest) != dests.end());
                ++chunkId;
            }
        }
    }
}

TEST(AllToAllTest, SingleNPU) {
    const int npusCount = 1;
    const int collectivesCount = 1;
    
    AllToAll alltoall(npusCount, collectivesCount);
    
    // With 1 NPU, there's 1 chunk: NPU 0 sends to NPU 0
    EXPECT_EQ(alltoall.chunksCount(), 1);
    EXPECT_EQ(alltoall.precondition(0), 0);
    
    const auto& dests = alltoall.postcondition(0);
    EXPECT_EQ(dests.size(), 1);
    EXPECT_TRUE(dests.find(0) != dests.end());
}

TEST(AllToAllTest, CompareWithAllGather) {
    const int npusCount = 4;
    
    // AllGather: N chunks, each goes to all N NPUs
    // Total communication: N * N transfers
    
    // AllToAll: N*N chunks, each goes to 1 NPU
    // Total communication: N * N transfers
    
    // Both have same number of transfers, but different patterns
    AllToAll alltoall(npusCount, 1);
    
    // AllToAll has more chunks but simpler destinations
    EXPECT_EQ(alltoall.chunksCount(), 16);  // 4*4 chunks
    
    // Each chunk has only 1 destination
    for (int chunk = 0; chunk < alltoall.chunksCount(); ++chunk) {
        EXPECT_EQ(alltoall.postcondition(chunk).size(), 1);
    }
}
