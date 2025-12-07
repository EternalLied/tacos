/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <optional>
#include <random>
#include <set>
#include <tacos/collective/collective.h>
#include <tacos/event_queue/event_queue.h>
#include <tacos/synthesizer/time_expanded_network.h>
#include <tacos/log.h>
#include <tacos/topology/topology.h>
#include <unordered_map>
#include <unordered_set>

namespace tacos {

/// @brief TACOS collective_ synthesizer.
class Synthesizer {
  public:
    // data types
    using Time = EventQueue::Time;
    using NpuID = Topology::NpuID;
    using ChunkID = Collective::ChunkID;
    using ChunkSize = Collective::ChunkSize;

    /// @brief Default constructor for the synthesizer.
    Synthesizer() noexcept;

    /// @brief Run TACOS synthesis process to synthesize a collective algorithm.
    /// @param topology Target network topology
    /// @param collective Target collective pattern
    /// @param chunkSize Size of each chunk (in bytes)
    /// @return estimated time to complete the collective pattern
    [[nodiscard]] Time solve(const Topology& topology,
                             const Collective& collective,
                             ChunkSize chunkSize) noexcept;

    /// @brief Calculate link utilization statistics (excluding switch-to-switch links)
    /// @param topology Target network topology
    /// @return average link utilization as a percentage (0-100)
    [[nodiscard]] double calculateLinkUtilization(const Topology& topology) const noexcept;

    /// @brief Multi-round synthesis result
    struct MultiRoundResult {
        std::unique_ptr<Synthesizer> bestSynthesizer;  // Use unique_ptr to avoid incomplete type issue
        Time bestCollectiveTime;
        int bestRound;
        int totalRounds;
        double totalSynthesisTime;  // in microseconds
        bool interrupted;
    };

    /// @brief Run multiple rounds of synthesis to find best result
    /// @param topology Target network topology
    /// @param collective Target collective pattern
    /// @param chunkSize Size of each chunk (in bytes)
    /// @param maxNoImprovementRounds Stop after this many rounds without improvement
    /// @param interruptFlag Atomic flag to check for user interruption
    /// @return Multi-round synthesis result
    [[nodiscard]] static MultiRoundResult solveMultiRound(
        const Topology& topology,
        const Collective& collective,
        ChunkSize chunkSize,
        int maxNoImprovementRounds,
        const std::atomic<bool>& interruptFlag) noexcept;

  private:
    /// @brief Map of destination NPU -> set of unsatisfied chunk IDs.
    using PostconditionMap = std::unordered_map<NpuID, std::unordered_set<ChunkID>>;

    /// @brief A condition: (chunkID, NpuID) pair
    using Condition = std::pair<ChunkID, NpuID>;

    /// @brief Target network topology.
    const Topology* topology_ = nullptr;

    /// @brief Target collective pattern to synthesize.
    const Collective* collective_ = nullptr;
    
    /// @brief Type of the collective being synthesized
    CollectiveType collectiveType_ = CollectiveType::UNKNOWN;

    /// @brief Number of NPUs in the topology.
    int npusCount = -1;

    /// @brief Number of chunks in the collective pattern.
    int chunksCount_ = -1;

    /// @brief Current time in the synthesizer.
    Time currentTime_ = -1;

    /// @brief Synthesized collective time.
    Time collectiveTime_ = -1;

    /// @brief Event queue to manage synthesizer events.
    EventQueue eventQueue_ = {};

    /// @brief Time-expanded network to model the topology over time.
    std::unique_ptr<TimeExpandedNetwork> ten_ = nullptr;

    /// @brief true if chunk c has arrived at NPU n: chunkMap_[c][n] = true
    std::vector<std::vector<bool>> chunkMap_ = {};

    /// @brief Pre-sorted postcondition list (generated once, updated each iteration)
    std::vector<Condition> sortedPostconditions_ = {};
    
    /// @brief Random number generator engine
    std::mt19937 randomEngine{std::random_device{}()};
    
#ifdef ENABLE_PERF_STATS
    /// @brief Performance statistics: time spent in priority adjustment (microseconds)
    mutable double priorityAdjustmentTime_ = 0.0;
    
    /// @brief Performance statistics: number of priority adjustments performed
    mutable int priorityAdjustmentCount_ = 0;
#endif

    /// @brief Initialize the synthesizer with the given topology and collective.
    /// @param topology target network topology
    /// @param collective target collective pattern
    /// @param chunkSize chunk size in bytes
    void initialize_(const Topology& topology,
                     const Collective& collective,
                     ChunkSize chunkSize) noexcept;

    /// @brief Mark chunks in precondition as already at their source NPUs.
    void markPrecondition_() noexcept;

    /// @brief Initialize and sort all postconditions once (called after markPrecondition_)
    void initializeSortedPostconditions_() noexcept;
    
    /// @brief Remove satisfied conditions from sortedPostconditions_ (in-place)
    void pruneSatisfiedPostconditions_() noexcept;

    /// @brief Dynamically adjust priority of a single postcondition based on updated distance
    /// @param chunk chunk ID that had a state update
    /// @param dest destination NPU ID
    void adjustPostconditionPriority_(ChunkID chunk, NpuID dest) noexcept;

    /// @brief Expand the TEN to the updated current timetsep.
    /// @details During the expansion, the TEN will check which chunk arrived
    /// at the designated destination NPUs, check the replacement opportunities,
    /// and update the chunkMap_ accordingly.
    /// @param postconditionMap map of unsatisfied postconditions
  /// @brief Expand the TEN for the current timestep and return statistics
  /// @return pair: <replacements_performed, discarded_transfers>
  [[nodiscard]] std::pair<int, int> expandTenTimestep_(PostconditionMap* postconditionMap) noexcept;

    /// @brief Find a replacement candidate for a matched link-chunk transfer.
    /// @details For heterogeneous networks, a chunk may have arrived the destination NPU
    /// already by leveraging a shorter path.
    /// If so, instead of simply discarding the link-chunk matching,
    /// TACOS will try to find another chunk that's not satisfied yet, to maximize
    /// the network utilization.
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @param postconditionMap map of unsatisfied postconditions
    /// @return a replacement chunk ID if found, std::nullopt otherwise
    [[nodiscard]] std::optional<ChunkID> findReplacementChunk_(
        NpuID src, NpuID dest, const PostconditionMap* postconditionMap) noexcept;

    /// @brief Make a link-chunk matching for a given chunk and destination NPU.
    /// @details This method iterates through all source NPUs that have the chunk,
    /// applies policy filters (AllGather: direct neighbors only; AllToAll: greedy routing),
    /// and selects the best candidate source NPU to transfer the chunk
    /// @param chunk chunk ID to transfer
    /// @param dest destination NPU ID
    /// @return the selected source NPU (-1 if failed)
    int linkChunkMatching_(ChunkID chunk, NpuID dest) noexcept;

    /// @brief Compare lhs and rhs Time values for equality
    /// @param lhs Time value
    /// @param rhs Time value
    /// @return true if lhs and rhs are equal (epsilon 1e-9), false otherwise
    [[nodiscard]] static bool isEqual(Time lhs, Time rhs) noexcept;
    
    /// @brief Try AllToAll greedy routing for a candidate source
    /// @param chunk chunk ID to transfer
    /// @param selectedSrc candidate source NPU
    /// @param dest final destination NPU
    /// @return true if routing succeeded, false otherwise
    bool tryAllToAllRouting_(ChunkID chunk, NpuID selectedSrc, NpuID dest) noexcept;
    
    /// @brief Try AllGather direct routing (precomputed routes)
    /// @param chunk chunk ID to transfer
    /// @param selectedSrc candidate source NPU
    /// @param dest destination NPU
    /// @return true if routing succeeded, false otherwise
    bool tryAllGatherRouting_(ChunkID chunk, NpuID selectedSrc, NpuID dest) noexcept;
};
}  // namespace tacos
