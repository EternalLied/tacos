/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <iostream>
#include <iomanip>
#include <cassert>
#include <limits>
#include <algorithm>
#include <map>
#include <chrono>
#include <tacos/synthesizer/synthesizer.h>
#include <tacos/event_queue/timer.h>
#include "log.h"

using namespace tacos;

Synthesizer::Synthesizer() noexcept = default;

Synthesizer::Time Synthesizer::solve(const Topology& topology,
                                     const Collective& collective,
                                     ChunkSize chunkSize) noexcept {
    assert(chunkSize > 0);

    // Performance tracking variables (only active if ENABLE_PERF_STATS is defined)
#ifdef ENABLE_PERF_STATS
    auto solveTimer = Timer();
    solveTimer.start();
    auto initTimer = Timer();
    
    int iterationCount = 0;
    double totalFilterTime = 0.0;
    double totalExpandTime = 0.0;
    double totalPruneTime = 0.0;
    double totalMatchingTime = 0.0;
    auto perfTimer = Timer();
    
    // Track initialization phase
    initTimer.start();
#endif

    // initialize the synthesizer
    initialize_(topology, collective, chunkSize);

    // mark trivial initial case
    // that is, chunks in preconditions are already at their sources
    markPrecondition_();

    // Initialize and sort all postconditions ONCE
    initializeSortedPostconditions_();

#ifdef ENABLE_PERF_STATS
    initTimer.stop();
    auto totalInitTime = initTimer.time();
#endif    // then, repeat the link-chunk matching process
    while (!eventQueue_.empty()) {
        PerfLog(++iterationCount);
        
        // get current event time
        currentTime_ = eventQueue_.pop();
        DebugLog(std::cout << "[TacosEvent]" << std::endl);
        DebugLog(std::cout << "At Time: " << currentTime_ << std::endl);

        // === PHASE 1: Convert sorted postconditions to map (for replacement logic) ===
        PerfLog(perfTimer.start());
        auto postconditionMap = PostconditionMap();
        for (const auto& [chunk, dest] : sortedPostconditions_) {
            postconditionMap[dest].insert(chunk);
        }
        PerfLog(perfTimer.stop(); totalFilterTime += perfTimer.time());

        // === PHASE 2: Expand TEN ===
        PerfLog(perfTimer.start());
        const auto [replacedCount, discardedCount] = expandTenTimestep_(&postconditionMap);
        PerfLog(perfTimer.stop(); totalExpandTime += perfTimer.time());
        DebugLog(std::cout << "Replaced: " << replacedCount << std::endl);
        DebugLog(std::cout << "Discarded: " << discardedCount << std::endl);

        // === PHASE 3: Prune Satisfied Postconditions ===
        // Remove conditions that have been satisfied (much faster than re-sorting)
        PerfLog(perfTimer.start());
        pruneSatisfiedPostconditions_();
        PerfLog(perfTimer.stop(); totalPruneTime += perfTimer.time());

        DebugLog(std::cout << "unsatisfied postconditions: " << sortedPostconditions_.size() << std::endl);

        if (sortedPostconditions_.empty()) {
            // no unsatisfied postcondition left to map
            // if so, just proceed to the next event
            // until all chunks arrive at their destinations
            continue;
        }

        auto successfulMatchingCount = 0;
        std::vector<std::tuple<ChunkID, NpuID, NpuID, std::vector<int>>> matchedRoutes;

        // Clear round-used edges at the start of each timestep matching
        // This ensures no physical edge is used by multiple routes in the same round
        ten_->clearRoundUsedEdges();

        // Optimization: Cache backtrack results for this timestep
        // Multiple conditions may have the same destination, so we cache to avoid repeated computation
        std::unordered_map<NpuID, std::unordered_set<NpuID>> backtrackCache;

        // Matching policy (per-route, no global hop limit):
        // - If shortest path has no switch, require 1-hop only
        // - If path includes switches, prioritize fewer hops (no global cap)
        // - Reject routes mixing direct device edges with switch edges
        // All policy logic is implemented inside linkChunkMatching_

        // === PHASE 4: Link-Chunk Matching ===
        PerfLog(perfTimer.start());
        for (const auto [chunk, dest] : sortedPostconditions_) {
            // linkChunkMatching_ returns the selected source NPU, or -1 if failed
            const auto selectedSrc = linkChunkMatching_(chunk, dest, backtrackCache);
            
            if (selectedSrc >= 0) {
                // Get the actual route path that was reserved
                auto path = ten_->getRoutePath(selectedSrc, dest);
                matchedRoutes.push_back({chunk, selectedSrc, dest, path});
                ++successfulMatchingCount;
            }
        }
        PerfLog(perfTimer.stop(); totalMatchingTime += perfTimer.time());
        DebugLog(std::cout << "Scheduled: " << successfulMatchingCount << std::endl);
        
        // Print matched routes with physical hop count prioritized
        DebugLog(
            int totalLogicalHops = 0;
            int totalPhysicalHops = 0;
            for (const auto& [chunk, src, dest, path] : matchedRoutes) {
                // Physical hops = path length - 1
                int physicalHops = path.empty() ? 0 : static_cast<int>(path.size()) - 1;
                // Logical hops from TEN (considers Cut-Through switches)
                int logicalHops = ten_->routeHopCount(src, dest);
                if (logicalHops < 0) logicalHops = physicalHops; // fallback
                
                totalLogicalHops += logicalHops;
                totalPhysicalHops += physicalHops;
                
                std::cout << "  Chunk " << chunk << ": GPU" << src << " -> GPU" << dest 
                          << " [" << physicalHops << " physical hop" << (physicalHops != 1 ? "s" : "") << "]";
                if (logicalHops != physicalHops) {
                    std::cout << " (" << logicalHops << " logical)";
                }
                if (!path.empty()) {
                    std::cout << " (";
                    for (size_t i = 0; i < path.size(); ++i) {
                        std::cout << path[i];
                        if (i < path.size() - 1) std::cout << "->";
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;
            }
            if (!matchedRoutes.empty()) {
                double avgLogical = static_cast<double>(totalLogicalHops) / matchedRoutes.size();
                double avgPhysical = static_cast<double>(totalPhysicalHops) / matchedRoutes.size();
                std::cout << "  Average physical hops: " << std::fixed << std::setprecision(2) << avgPhysical;
                if (avgLogical != avgPhysical) {
                    std::cout << " (logical: " << avgLogical << ")";
                }
                std::cout << std::endl;
            }
        );
        DebugLog(std::cout << std::endl);
    }

    // Check if all postconditions were satisfied
    if (!sortedPostconditions_.empty()) {
        std::cerr << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "ERROR: Scheduling failed!" << std::endl;
        std::cerr << "Event queue is empty but " << sortedPostconditions_.size() 
                  << " postcondition(s) remain unsatisfied." << std::endl;
        std::cerr << "Collective type: " << (collectiveType_ == CollectiveType::ALL_GATHER ? "AllGather" : 
                                              collectiveType_ == CollectiveType::ALL_TO_ALL ? "AllToAll" : "Unknown") << std::endl;
        std::cerr << std::endl;
        std::cerr << "Possible causes:" << std::endl;
        std::cerr << "  1. Topology connectivity: Some NPU pairs are not reachable" << std::endl;
        std::cerr << "  2. Routing policy: Current policy blocks required routes" << std::endl;
        if (collectiveType_ == CollectiveType::ALL_GATHER) {
            std::cerr << "  3. AllGather policy: Only allows 1-hop routes in non-switch topologies" << std::endl;
        }
        std::cerr << std::endl;
        std::cerr << "Unsatisfied postconditions (first 10):" << std::endl;
        int count = 0;
        for (const auto& [chunk, dest] : sortedPostconditions_) {
            if (count >= 10) break;
            auto src = collective_->precondition(chunk);
            std::cerr << "  Chunk " << chunk << ": GPU" << src << " -> GPU" << dest << std::endl;
            ++count;
        }
        std::cerr << "========================================" << std::endl;
        std::cerr << std::endl;
        return -1;  // Return error value
    }

    // === Print Performance Summary ===
    PerfLog(
        solveTimer.stop();
        auto totalSolveTime = solveTimer.time();
        
        double totalLoopTime = totalFilterTime + totalExpandTime + totalPruneTime + totalMatchingTime;
        std::cout << std::endl;
        std::cout << "=================================================" << std::endl;
        std::cout << "======== Synthesizer Performance Summary ========" << std::endl;
        std::cout << "Total solve() time: " << totalSolveTime / 1000.0 << " ms" << std::endl;
        std::cout << "Total iterations: " << iterationCount << std::endl;
        std::cout << std::endl;
        
        std::cout << "Time breakdown:" << std::endl;
        std::cout << "  Initialization:          " << std::fixed << std::setprecision(2) << std::setw(8)
                  << totalInitTime / 1000.0 << " ms  (" << std::setw(5)
                  << (totalInitTime / totalSolveTime * 100) << "%)" << std::endl;
        std::cout << "  convertToMap:            " << std::setw(8)
                  << totalFilterTime / 1000.0 << " ms  (" << std::setw(5)
                  << (totalFilterTime / totalSolveTime * 100) << "%)" << std::endl;
        std::cout << "  expandTenTimestep:       " << std::setw(8)
                  << totalExpandTime / 1000.0 << " ms  (" << std::setw(5)
                  << (totalExpandTime / totalSolveTime * 100) << "%)" << std::endl;
        std::cout << "  pruneSatisfiedPostcond:  " << std::setw(8)
                  << totalPruneTime / 1000.0 << " ms  (" << std::setw(5)
                  << (totalPruneTime / totalSolveTime * 100) << "%)" << std::endl;
        std::cout << "  linkChunkMatching:       " << std::setw(8)
                  << totalMatchingTime / 1000.0 << " ms  (" << std::setw(5)
                  << (totalMatchingTime / totalSolveTime * 100) << "%)" << std::endl;
        std::cout << "=================================================" << std::endl;
        std::cout << std::endl;
    );

    // all matching has been finished
    // return measured collective_ time
    assert(collectiveTime_ > 0);
    return collectiveTime_;
}

void Synthesizer::initialize_(const Topology& topology,
                              const Collective& collective,
                              const ChunkSize chunkSize) noexcept {
    // reset the event queue
    eventQueue_.reset();
    currentTime_ = 0;

    // set topology and collective
    topology_ = &topology;
    collective_ = &collective;
    collectiveType_ = collective.getType();

    // set variables
    npusCount = topology_->npusCount();
    chunksCount_ = collective_->chunksCount();

    // construct TEN from the topology
    ten_ = std::make_unique<TimeExpandedNetwork>(*topology_, chunkSize);

    // construct chunkMap_
    chunkMap_.assign(chunksCount_, std::vector<bool>(npusCount, false));
}

void Synthesizer::markPrecondition_() noexcept {
    // for every chunk, mark its source NPU as true in the chunkMap_
    for (auto chunk = 0; chunk < chunksCount_; ++chunk) {
        const auto src = collective_->precondition(chunk);
        chunkMap_[chunk][src] = true;
    }
}

void Synthesizer::initializeSortedPostconditions_() noexcept {
    // Build initial postcondition map
    auto postconditionMap = PostconditionMap();
    for (auto chunk = 0; chunk < chunksCount_; ++chunk) {
        const auto dests = collective_->postcondition(chunk);
        for (const auto dest : dests) {
            if (!chunkMap_[chunk][dest]) {
                postconditionMap[dest].insert(chunk);
            }
        }
    }
    
    // Sort by physical hop count (same logic as shufflePostcondition_)
    std::unordered_map<NpuID, std::unordered_set<NpuID>> backtrackCache;
    
    auto computeMinPhysicalHops = [this, &backtrackCache](ChunkID chunk, NpuID dest) -> int {
        int minHops = std::numeric_limits<int>::max();
        auto it = backtrackCache.find(dest);
        if (it == backtrackCache.end()) {
            it = backtrackCache.emplace(dest, ten_->backtrack(dest)).first;
        }
        const auto& sources = it->second;
        for (const auto src : sources) {
            if (chunkMap_[chunk][src]) {
                int hops = ten_->getRoutePhysicalHops(src, dest);
                minHops = std::min(minHops, hops);
            }
        }
        return minHops;
    };
    
    // Flatten and compute hop counts
    std::vector<std::pair<Condition, int>> conditionsWithHops;
    conditionsWithHops.reserve(postconditionMap.size() * 8);
    for (const auto& [dest, chunks] : postconditionMap) {
        for (const auto chunk : chunks) {
            int minPhysicalHops = computeMinPhysicalHops(chunk, dest);
            conditionsWithHops.emplace_back(Condition{chunk, dest}, minPhysicalHops);
        }
    }
    
    // Sort by physical hop count
    // Strategy depends on collective type:
    // - AllGather: Ascending (short paths first) - no intermediate caching needed
    // - AllToAll: Descending (long paths first) - benefits from progressive intermediate caching
    const bool useDescendingOrder = (collectiveType_ == CollectiveType::ALL_TO_ALL);
    
    std::stable_sort(conditionsWithHops.begin(), conditionsWithHops.end(),
        [useDescendingOrder](const auto& a, const auto& b) {
            return useDescendingOrder ? (a.second > b.second) : (a.second < b.second);
        });
    
    // Shuffle within same-hop groups
    for (auto it = conditionsWithHops.begin(); it != conditionsWithHops.end(); ) {
        int currentHops = it->second;
        auto groupEnd = std::find_if(it, conditionsWithHops.end(),
            [currentHops](const auto& elem) { 
                return elem.second != currentHops; 
            });
        std::shuffle(it, groupEnd, randomEngine);
        it = groupEnd;
    }
    
    // Extract sorted conditions
    sortedPostconditions_.clear();
    sortedPostconditions_.reserve(conditionsWithHops.size());
    for (const auto& [cond, _] : conditionsWithHops) {
        sortedPostconditions_.push_back(cond);
    }
}

void Synthesizer::pruneSatisfiedPostconditions_() noexcept {
    // Remove conditions that have been satisfied (in-place removal)
    sortedPostconditions_.erase(
        std::remove_if(sortedPostconditions_.begin(), sortedPostconditions_.end(),
            [this](const Condition& cond) {
                return chunkMap_[cond.first][cond.second];
            }),
        sortedPostconditions_.end()
    );
}

Synthesizer::PostconditionMap Synthesizer::filterPostcondition_() const noexcept {
    auto postconditionMap = PostconditionMap();

    // iterate over all chunks
    for (auto chunk = 0; chunk < chunksCount_; ++chunk) {
        // check which destination NPUs have not yet received the chunk
        const auto dests = collective_->postcondition(chunk);
        for (const auto dest : dests) {
            if (!chunkMap_[chunk][dest]) {
                postconditionMap[dest].insert(chunk);
            }
        }
    }

    return postconditionMap;
}

std::vector<Synthesizer::Condition> Synthesizer::shufflePostcondition_(
    const PostconditionMap& postconditionMap) noexcept {
    
    // Optimization: Cache backtrack results per destination (significant speedup for large postconditions)
    std::unordered_map<NpuID, std::unordered_set<NpuID>> backtrackCache;
    
    // Helper function to compute minimum physical hop count for a condition
    // Use physical hops for sorting to avoid Cut-Through routes getting unfair priority
    auto computeMinPhysicalHops = [this, &backtrackCache](ChunkID chunk, NpuID dest) -> int {
        int minHops = std::numeric_limits<int>::max();
        
        // Get or compute backtrack sources for this destination
        auto it = backtrackCache.find(dest);
        if (it == backtrackCache.end()) {
            it = backtrackCache.emplace(dest, ten_->backtrack(dest)).first;
        }
        const auto& sources = it->second;
        
        for (const auto src : sources) {
            if (chunkMap_[chunk][src]) {
                int hops = ten_->getRoutePhysicalHops(src, dest);
                minHops = std::min(minHops, hops);
            }
        }
        return minHops;
    };
    
    // Step 1: Flatten and precompute physical hop counts (compute once, use multiple times)
    std::vector<std::pair<Condition, int>> conditionsWithHops;
    conditionsWithHops.reserve(postconditionMap.size() * 8); // Rough estimate
    
    for (const auto& [dest, chunks] : postconditionMap) {
        for (const auto chunk : chunks) {
            int minPhysicalHops = computeMinPhysicalHops(chunk, dest);
            conditionsWithHops.emplace_back(Condition{chunk, dest}, minPhysicalHops);
        }
    }
    
    // Step 2: Sort by physical hop count (using precomputed values)
    // Strategy depends on collective type:
    // - AllGather: Ascending (short paths first) - no intermediate caching needed
    // - AllToAll: Descending (long paths first) - benefits from progressive intermediate caching
    const bool useDescendingOrder = (collectiveType_ == CollectiveType::ALL_TO_ALL);
    
    std::stable_sort(conditionsWithHops.begin(), conditionsWithHops.end(),
        [useDescendingOrder](const auto& a, const auto& b) {
            return useDescendingOrder ? (a.second > b.second) : (a.second < b.second);
        });
    
    // Step 3: Shuffle within same-hop groups (using precomputed values)
    for (auto it = conditionsWithHops.begin(); it != conditionsWithHops.end(); ) {
        int currentHops = it->second;
        
        // Find end of same-hop-count group
        auto groupEnd = std::find_if(it, conditionsWithHops.end(),
            [currentHops](const auto& elem) { 
                return elem.second != currentHops; 
            });
        
        // Shuffle within this group
        std::shuffle(it, groupEnd, randomEngine);
        it = groupEnd;
    }
    
    // Step 4: Extract final postcondition list
    std::vector<Condition> postcondition;
    postcondition.reserve(conditionsWithHops.size());
    
    // Debug: Print physical hop count distribution (using cached hop counts)
    std::map<int,int> hopCountDist;
    DebugLog(
        for (const auto& [cond, hops] : conditionsWithHops) {
            if (hops != std::numeric_limits<int>::max()) {
                hopCountDist[hops]++;
            }
        }
        if (!hopCountDist.empty()) {
            std::cout << "Postcondition physical hop distribution (ideal): ";
            for (const auto& [hops, count] : hopCountDist) {
                std::cout << hops << "-hop:" << count << " ";
            }
            std::cout << std::endl;
            std::cout << "  Note: Actual scheduled hops may be longer due to resource conflicts." << std::endl;
        }
    );
    
    for (const auto& [cond, _] : conditionsWithHops) {
        postcondition.push_back(cond);
    }
    
    return postcondition;
}

std::pair<int, int> Synthesizer::expandTenTimestep_(PostconditionMap* const postconditionMap) noexcept {
    // first, expand the TEN structure
    ten_->timestep(currentTime_);

    // this bool value is to track if any meaningful event happened during this timestep
    // e.g., chunk arrival or replacement
    // so that we can update the collective time
    auto eventHappened = false;

    int replacedCount = 0;
    int discardedCount = 0;

    // for every src-dest pairs
    for (auto src = 0; src < npusCount; src++) {
        for (auto dest = 0; dest < npusCount; dest++) {
            // if TEN is not available, skip
            // i.e., link doesn't exist or it is busy transferring a chunk
            if (!ten_->available(src, dest)) {
                continue;
            }

            // if a TEN link is available, there are two cases:
            // 1. the TEN link is indeed free, or
            // 2. it just become free by finishing a transfer
            // for case 2, we should mark this transfer as finished
            // and check for replacement possibilities

            // for case 1 (link is free), we can skip this
            auto chunk = ten_->chunk(src, dest);
            if (chunk < 0) {
                continue;
            }

            // for case 2, check if the chunk has already arrived at dest
            // by following other paths
            // and if so, check if we can replace this path with another chunk
            if (chunkMap_[chunk][dest]) {
                // Check if dest is actually a final destination for this chunk
                // For AllToAll: dest might just be an intermediate node, not the final target
                // For AllGather: all nodes are final destinations
                const auto& postconditions = collective_->postcondition(chunk);
                const bool isActualDestination = std::find(
                    postconditions.begin(), 
                    postconditions.end(), 
                    dest
                ) != postconditions.end();
                
                if (isActualDestination) {
                    // dest is a final destination and has already received this chunk
                    // so we can replace it with another chunk that dest needs
                    const auto replacementChunk = findReplacementChunk_(src, dest, postconditionMap);

                    if (!replacementChunk.has_value()) {
                        // no replacement candidate found
                        // just mark this TEN link as available and skip
                        ten_->transferFinished(src, dest);
                        ++discardedCount;
                        continue;
                    }

                    // replacement candidate found
                    chunk = replacementChunk.value();
                    ++replacedCount;
                } else {
                    // dest is just an intermediate node (for AllToAll multi-hop routing)
                    // The chunk should continue to its final destination
                    // Do NOT replace it - let it proceed normally
                }
            }

            // a meaningful chunk (regardless of replacement) has arrived at dest
            eventHappened = true;

            // mark the chunk arrived at dest, and mark this TEN link as available
            chunkMap_[chunk][dest] = true;
            ten_->transferFinished(src, dest);

            // mark this postcondition as satisfied
            // i.e., remove this chunk from the postcondition map
            auto it = postconditionMap->find(dest);
            if (it != postconditionMap->end()) {
                it->second.erase(chunk);
                if (it->second.empty()) {
                    postconditionMap->erase(it);
                }
            }
        }
    }

    // at the end of the TEN expansion
    // if a meaningful event happened, we should update the collective time
    if (eventHappened) {
        // update collective time to current time
        collectiveTime_ = currentTime_;
    }

    return {replacedCount, discardedCount};
}

std::optional<Synthesizer::ChunkID> Synthesizer::findReplacementChunk_(
    const NpuID src, const NpuID dest, const PostconditionMap* const postconditionMap) noexcept {
    // trivial scenario: if dest has all postconditions satisfied,
    // there's no need for replacement
    if (postconditionMap->find(dest) == postconditionMap->end()) {
        return std::nullopt;
    }

    // check if any replacement candidate exists
    auto candidates = std::vector<ChunkID>();

    // iterate over all unsatisfied postcondition of this dest NPU
    for (const auto chunk : postconditionMap->at(dest)) {
        // if this chunk is available at src NPU
        // but has not yet arrived at dest NPU,
        // this chunk can be a replacement candidate
        if (chunkMap_[chunk][src] && !chunkMap_[chunk][dest]) {
            candidates.push_back(chunk);
        }
    }

    // if there's no candidate, return nullopt
    if (candidates.empty()) {
        return std::nullopt;
    }

    // if there's only one candidate, return it
    if (candidates.size() == 1) {
        return candidates[0];
    }

    // if there are multiple candidates, randomly select one and return it
    auto dist = std::uniform_int_distribution<>(0, candidates.size() - 1);
    const auto idx = dist(randomEngine);
    return candidates[idx];
}

int Synthesizer::linkChunkMatching_(const ChunkID chunk, const NpuID dest,
                                   std::unordered_map<NpuID, std::unordered_set<NpuID>>& backtrackCache) noexcept {
    // Get or compute backtrack source NPUs (use cache for performance)
    auto it = backtrackCache.find(dest);
    if (it == backtrackCache.end()) {
        it = backtrackCache.emplace(dest, ten_->backtrack(dest)).first;
    }
    const auto& sources = it->second;

    // filter candidate link-chunk matching
    // prioritize by: 1) minimum physical hop count, 2) earliest arrival time
    // Use physical hops to avoid Cut-Through routes getting unfair priority
    auto minHopCount = std::numeric_limits<int>::max();
    auto arrivalTime = std::numeric_limits<Time>::max();
    auto candidates = std::vector<NpuID>();

    // iterate over all source NPUs
    for (const auto src : sources) {
        // if src does not have the chunk, skip
        if (!chunkMap_[chunk][src]) {
            continue;
        }

        // get physical hop count and transfer time for this route
        const auto hopCount = ten_->getRoutePhysicalHops(src, dest);
        const auto linkWeight = ten_->linkTransferTime(src, dest);
        const auto linkTime = currentTime_ + linkWeight;

        // skip invalid routes
        if (hopCount < 0) continue;

        // Policy enforcement (per-route) - different strategies for different collectives
        const bool hasSwitchInTopology = ten_->hasSwitches();
        const bool routeThroughSwitch = ten_->routeHasSwitch(src, dest);
        const bool hasMixedEdges = ten_->routeHasMixedEdges(src, dest);
        
        if (collectiveType_ == CollectiveType::ALL_GATHER) {
            // === AllGather: Strict routing policy (original) ===
            // 1) If topology has no switches, only schedule 1-hop routes
            if (!hasSwitchInTopology && hopCount != 1) continue;

            // 2) If the route's shortest path does not traverse any switch, require 1-hop
            if (!routeThroughSwitch && hopCount != 1) continue;

            // 3) Do not match routes that mix direct device edges with switch edges
            if (hasMixedEdges) continue;
            
        } else if (collectiveType_ == CollectiveType::ALL_TO_ALL) {
            // === AllToAll: Relaxed routing policy ===
            // Allow multi-hop routes and mixed edges for better connectivity
            // This is necessary for topologies like mesh where not all NPU pairs are directly connected
            // No restrictions on hop count or edge mixing
            
        } else {
            // === Unknown collective: Use conservative AllGather policy ===
            if (!hasSwitchInTopology && hopCount != 1) continue;
            if (!routeThroughSwitch && hopCount != 1) continue;
            if (hasMixedEdges) continue;
        }

        // Priority 1: Prefer routes with fewer hops (better link utilization)
        if (hopCount < minHopCount) {
            minHopCount = hopCount;
            arrivalTime = linkTime;
            candidates.clear();
            candidates.emplace_back(src);
        } else if (hopCount == minHopCount) {
            // Priority 2: Among same hop count, prefer earliest arrival
            if (linkTime < arrivalTime) {
                arrivalTime = linkTime;
                candidates.clear();
                candidates.emplace_back(src);
            } else if (isEqual(linkTime, arrivalTime)) {
                candidates.emplace_back(src);
            }
        }
    }

    // if candidates are empty, no match can be made
    if (candidates.empty()) {
        return -1;
    }

    // randomly shuffle and select one source NPU to make link-chunk match
    // (among candidates with same hop count and arrival time)
    std::shuffle(candidates.begin(), candidates.end(), randomEngine);
    
    // Try candidates in shuffled order until we find one with an available physical route
    // (routes may become unavailable between backtrack() call and actual reservation)
    for (const auto selectedSrc : candidates) {
        // === AllToAll multi-hop optimization ===
        // For AllToAll with multi-hop routes crossing intermediate devices,
        // only reserve the segment to the first intermediate device to avoid
        // occupying too many links in a single timestep
        if (collectiveType_ == CollectiveType::ALL_TO_ALL) {
            const int intermediateDevice = ten_->getFirstIntermediateDevice(selectedSrc, dest);
            
            if (intermediateDevice >= 0) {
                // Multi-hop route with intermediate device found
                // Check only the partial route (src -> intermediate) availability
                if (!ten_->canReserveRoute(selectedSrc, intermediateDevice)) {
                    continue;  // Partial route not available, try next candidate
                }
                
                // Reserve only src -> intermediate segment
                const auto intermediateArrivalTime = currentTime_ + ten_->linkTransferTime(selectedSrc, intermediateDevice);
                
                // DebugLog(
                //     std::cout << "  [DEBUG] AllToAll partial: Chunk " << chunk 
                //               << " GPU" << selectedSrc << " -> GPU" << intermediateDevice 
                //               << " (intermediate, final dest: GPU" << dest << ")" << std::endl;
                // );
                
                ten_->transferChunkPartial(selectedSrc, intermediateDevice, chunk, intermediateArrivalTime);
                eventQueue_.schedule(intermediateArrivalTime);
                
                // Update chunkMap to reflect chunk now at intermediate device
                // This allows the intermediate device to serve as a source for subsequent hops
                chunkMap_[chunk][intermediateDevice] = true;
                
                return selectedSrc;
            }
        }
        
        // === Standard full route check (AllGather or direct routes) ===
        // Final check: verify physical route is still available
        if (!ten_->canReserveRoute(selectedSrc, dest)) {
            continue;  // Route no longer available, try next candidate
        }
        
        // Route is available - reserve it
        // DebugLog(
        //     std::cout << "  [DEBUG] At time " << currentTime_ << ", reserving route for Chunk " << chunk 
        //               << ": GPU" << selectedSrc << " -> GPU" << dest 
        //               << " (arrival: " << arrivalTime << ")" << std::endl;
        // );
        ten_->transferChunk(selectedSrc, dest, chunk, arrivalTime);

        // schedule an event when the matched chunk arrives
        eventQueue_.schedule(arrivalTime);

        return selectedSrc;
    }
    
    // All candidates' routes became unavailable
    return -1;
}

bool Synthesizer::isEqual(const Time lhs, const Time rhs) noexcept {
    constexpr Time epsilon = 1e-9;
    return std::abs(lhs - rhs) < epsilon;
}

double Synthesizer::calculateLinkUtilization(const Topology& topology) const noexcept {
    if (!ten_ || collectiveTime_ <= 0) {
        return 0.0;
    }
    
    const int npusCount = topology.npusCount();
    const auto& physLinks = topology.physLinks();
    
    // Calculate utilization for physical links (excluding switch-to-switch)
    double totalUtilization = 0.0;
    int linkCount = 0;
    
    for (const auto& link : physLinks) {
        const bool srcIsDevice = (link.src < npusCount);
        const bool dstIsDevice = (link.dst < npusCount);
        
        // Skip switch-to-switch links (both src and dst are switches)
        if (!srcIsDevice && !dstIsDevice) {
            continue;
        }
        
        // Get accumulated busy time for this physical edge
        double linkUtilization = ten_->getLinkUtilization(link.src, link.dst, collectiveTime_);
        
        totalUtilization += linkUtilization;
        linkCount++;
    }
    
    if (linkCount == 0) {
        return 0.0;
    }
    
    // Return average utilization as percentage
    return (totalUtilization / linkCount) * 100.0;
}

Synthesizer::MultiRoundResult Synthesizer::solveMultiRound(
    const Topology& topology,
    const Collective& collective,
    ChunkSize chunkSize,
    int maxNoImprovementRounds,
    const std::atomic<bool>& interruptFlag) noexcept {
    
    MultiRoundResult result;
    result.bestSynthesizer = nullptr;
    result.bestCollectiveTime = std::numeric_limits<Time>::max();
    result.bestRound = 0;
    result.totalRounds = 0;
    result.totalSynthesisTime = 0.0;
    result.interrupted = false;
    
    int noImprovementCount = 0;
    
    while (true) {
        ++result.totalRounds;
        
        if (interruptFlag.load()) {
            result.interrupted = true;
            break;
        }
        
        // Run one round of synthesis
        auto synthesizer = std::make_unique<Synthesizer>();
        
        // Measure synthesis time
        auto startTime = std::chrono::high_resolution_clock::now();
        auto collectiveTime = synthesizer->solve(topology, collective, chunkSize);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto roundTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        result.totalSynthesisTime += roundTime;
        
        // Check if this is the best result
        if (collectiveTime < result.bestCollectiveTime) {
            result.bestCollectiveTime = collectiveTime;
            result.bestRound = result.totalRounds;
            result.bestSynthesizer = std::move(synthesizer);
            noImprovementCount = 0;
        } else {
            noImprovementCount++;
            
            // Early stopping if no improvement for several rounds
            if (noImprovementCount >= maxNoImprovementRounds) {
                break;
            }
        }
    }
    
    return result;
}

