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
#include <tacos/log.h>

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

        // Matching policy (per-route, no global hop limit):
        // - AllGather: Only match direct neighbors (single-hop without crossing devices)
        // - AllToAll: Greedy routing with minimum distance source selection
        // - Reject routes mixing direct device edges with switch edges
        // All policy logic is implemented inside linkChunkMatching_

        // === PHASE 4: Link-Chunk Matching ===
        PerfLog(perfTimer.start());
        for (const auto [chunk, dest] : sortedPostconditions_) {
            // AllToAll-only: Skip if already scheduled (prevents duplicate scheduling)
            // AllGather uses replacement mechanism and benefits from re-scheduling
            if (collectiveType_ == CollectiveType::ALL_TO_ALL) {
                if (scheduledAllToAllPostconditions_.count({chunk, dest}) > 0) {
                    continue;  // Already scheduled, skip to prevent duplicate
                }
            }
            
            // linkChunkMatching_ returns the selected source NPU, or -1 if failed
            const auto selectedSrc = linkChunkMatching_(chunk, dest);
            
            if (selectedSrc >= 0) {
                // Get the actual route path that was reserved
                auto path = ten_->getRoutePath(selectedSrc, dest);
                matchedRoutes.push_back({chunk, selectedSrc, dest, path});
                ++successfulMatchingCount;
                
                // AllToAll-only: Mark as scheduled to prevent re-scheduling
                if (collectiveType_ == CollectiveType::ALL_TO_ALL) {
                    scheduledAllToAllPostconditions_.insert({chunk, dest});
                }
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
        
        // Report priority adjustment statistics
        if (collectiveType_ == CollectiveType::ALL_TO_ALL) {
            if (priorityAdjustmentCount_ > 0) {
                const double avgAdjustTime = priorityAdjustmentTime_ / priorityAdjustmentCount_;
                const double totalMs = priorityAdjustmentTime_ / 1000.0;
                const double percentage = (priorityAdjustmentTime_ / totalSolveTime) * 100.0;
                
                std::cout << "  Priority adjustment:     " << std::setw(8)
                          << totalMs << " ms  (" << std::setw(5)
                          << percentage << "%)" << std::endl;
                std::cout << "    -> " << priorityAdjustmentCount_ << " calls, avg "
                          << std::fixed << std::setprecision(2) << avgAdjustTime << " us per call" << std::endl;
            }
        }
        
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
    
#ifdef ENABLE_PERF_STATS
    // reset performance statistics
    priorityAdjustmentTime_ = 0.0;
    priorityAdjustmentCount_ = 0;
#endif

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
    
    // Sort by transfer time (using distanceMatrix_ from TEN)
    auto computeMinTransferTime = [this](ChunkID chunk, NpuID dest) -> Time {
        Time minTime = std::numeric_limits<Time>::max();
        // Iterate through all devices to find sources with the chunk
        for (int src = 0; src < npusCount; ++src) {
            if (src != dest && chunkMap_[chunk][src]) {
                Time transferTime = ten_->getDistance(src, dest);
                minTime = std::min(minTime, transferTime);
            }
        }
        return minTime;
    };
    
    // Flatten and compute transfer times
    std::vector<std::pair<Condition, Time>> conditionsWithTimes;
    conditionsWithTimes.reserve(postconditionMap.size() * 8);
    for (const auto& [dest, chunks] : postconditionMap) {
        for (const auto chunk : chunks) {
            Time minTransferTime = computeMinTransferTime(chunk, dest);
            conditionsWithTimes.emplace_back(Condition{chunk, dest}, minTransferTime);
        }
    }
    
    // Sort by transfer time
    // Strategy depends on collective type:
    // - AllGather: Ascending (short transfers first) - no intermediate caching needed
    // - AllToAll: Descending (long transfers first) - benefits from progressive intermediate caching
    const bool useDescendingOrder = (collectiveType_ == CollectiveType::ALL_TO_ALL);
    
    std::stable_sort(conditionsWithTimes.begin(), conditionsWithTimes.end(),
        [useDescendingOrder](const auto& a, const auto& b) {
            return useDescendingOrder ? (a.second > b.second) : (a.second < b.second);
        });
    
    // Shuffle within same-time groups (±epsilon tolerance)
    constexpr Time epsilon = 1e-6;
    for (auto it = conditionsWithTimes.begin(); it != conditionsWithTimes.end(); ) {
        Time currentTime = it->second;
        auto groupEnd = std::find_if(it, conditionsWithTimes.end(),
            [currentTime, epsilon](const auto& elem) {
                return std::abs(elem.second - currentTime) > epsilon;
            });
        std::shuffle(it, groupEnd, randomEngine);
        it = groupEnd;
    }
    
    // Extract sorted conditions
    sortedPostconditions_.clear();
    sortedPostconditions_.reserve(conditionsWithTimes.size());
    for (const auto& [cond, _] : conditionsWithTimes) {
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

void Synthesizer::adjustPostconditionPriority_(const ChunkID chunk, const NpuID dest) noexcept {
#ifdef ENABLE_PERF_STATS
    // Performance measurement: start timer
    auto startTime = std::chrono::high_resolution_clock::now();
#endif
    
    // Find the condition in sortedPostconditions_
    const Condition targetCond{chunk, dest};
    auto it = std::find(sortedPostconditions_.begin(), sortedPostconditions_.end(), targetCond);
    
    if (it == sortedPostconditions_.end()) {
        return;  // condition already satisfied or not found
    }
    
    const size_t oldIndex = std::distance(sortedPostconditions_.begin(), it);
    
    // Compute new priority (minimum remaining distance from any source that has the chunk)
    Time newDistance = std::numeric_limits<Time>::max();
    for (int src = 0; src < npusCount; ++src) {
        if (src != dest && chunkMap_[chunk][src]) {
            Time distance = ten_->getDistance(src, dest);
            newDistance = std::min(newDistance, distance);
        }
    }
    
    if (newDistance == std::numeric_limits<Time>::max()) {
        return;  // no valid source found
    }
    
    // Lambda to compute distance for a given condition
    auto computeDistance = [this](const Condition& cond) -> Time {
        Time minDist = std::numeric_limits<Time>::max();
        for (int src = 0; src < npusCount; ++src) {
            if (src != cond.second && chunkMap_[cond.first][src]) {
                minDist = std::min(minDist, ten_->getDistance(src, cond.second));
            }
        }
        return minDist;
    };
    
    // Determine sort order based on collective type
    const bool useDescendingOrder = (collectiveType_ == CollectiveType::ALL_TO_ALL);
    
    size_t newIndex = oldIndex;
    
    // Linear scan to find correct new position (tested to be faster than binary search)
    // For descending order: distance decreased – scan backward to find insertion point
    // For ascending order: distance decreased – scan forward to find insertion point
    // When distances are equal, use 50% probability to continue scanning for randomness
    
    std::uniform_int_distribution<int> coinFlip(0, 1);
    
    if (useDescendingOrder) {
        // Descending: larger distance = higher priority (earlier position)
        // Since we scheduled one hop, distance decreased – need to move backward
        // Scan backward from current position to find first element with distance < newDistance
        while (newIndex < sortedPostconditions_.size() - 1) {
            const auto& nextCond = sortedPostconditions_[newIndex + 1];
            Time nextDistance = computeDistance(nextCond);
            
            if (newDistance < nextDistance) {
                ++newIndex;  // move backward (lower priority)
            } else if (newDistance == nextDistance && coinFlip(randomEngine) == 1) {
                ++newIndex;  // 50% chance to continue when equal
            } else {
                break;  // found correct position
            }
        }
    } else {
        // Ascending: smaller distance = higher priority (earlier position)
        // Since we scheduled one hop, distance decreased – need to move forward
        // Scan forward from current position to find first element with distance > newDistance
        while (newIndex > 0) {
            const auto& prevCond = sortedPostconditions_[newIndex - 1];
            Time prevDistance = computeDistance(prevCond);
            
            if (newDistance < prevDistance) {
                --newIndex;  // move forward (higher priority)
            } else if (newDistance == prevDistance && coinFlip(randomEngine) == 1) {
                --newIndex;  // 50% chance to continue when equal
            } else {
                break;  // found correct position
            }
        }
    }
    
    // Move the condition if position changed
    if (newIndex != oldIndex) {
        sortedPostconditions_.erase(it);
        sortedPostconditions_.insert(sortedPostconditions_.begin() + newIndex, targetCond);
        
        DebugLog(
            std::cout << "  [Priority Adjust] Chunk " << chunk << " -> GPU" << dest
                      << ": position " << oldIndex << " -> " << newIndex
                      << " (distance: " << newDistance << "μs)" << std::endl;
        );
    }
    
#ifdef ENABLE_PERF_STATS
    // Performance measurement: end timer and accumulate
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    priorityAdjustmentTime_ += duration.count();
    priorityAdjustmentCount_++;
#endif
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

    // === PHASE 1: Process chunk arrivals at current time ===
    // Get all chunks that arrive at this timestep
    auto arrivals = ten_->getArrivalsAt(currentTime_);
    
    // === PHASE 2: Process arrivals with replacement optimization (AllGather only) ===
    // AllToAll does not need replacement since each chunk sends different data to different destinations
    for (const auto& [chunk, src, dest] : arrivals) {
        auto finalChunk = chunk;
        
        // AllGather-only: Replacement optimization for duplicate arrivals
        if (collectiveType_ != CollectiveType::ALL_TO_ALL && chunkMap_[chunk][dest]) {
            // Check if dest is actually a final destination for this chunk
            const auto& postconditions = collective_->postcondition(chunk);
            const bool isActualDestination = std::find(
                postconditions.begin(),
                postconditions.end(),
                dest
            ) != postconditions.end();
            
            if (isActualDestination) {
                // dest is a final destination and already has this chunk
                // Try to find a replacement chunk that dest still needs
                const auto replacementChunk = findReplacementChunk_(src, dest, postconditionMap);
                
                if (replacementChunk.has_value()) {
                    // Found a replacement - use it instead
                    finalChunk = replacementChunk.value();
                    ++replacedCount;
                    
                    DebugLog(std::cout << "Replaced: Chunk " << chunk << " -> " << finalChunk 
                             << " at GPU" << dest << std::endl);
                } else {
                    // No replacement found - this arrival is wasted
                    ++discardedCount;
                    
                    DebugLog(std::cout << "Discarded: Chunk " << chunk << " at GPU" << dest 
                             << " (already arrived)" << std::endl);
                    continue;  // Don't mark anything
                }
            }
            // If not actual destination (intermediate node), process normally
        }
        
        // Mark the final chunk as arrived at destination
        chunkMap_[finalChunk][dest] = true;
        eventHappened = true;
        
        // Check if dest is the final destination for this chunk
        const auto& postconditions = collective_->postcondition(finalChunk);
        const bool isFinalDestination = std::find(
            postconditions.begin(),
            postconditions.end(),
            dest
        ) != postconditions.end();
        
        if (isFinalDestination) {
            // This is a final destination - postcondition satisfied
            // Update postcondition map
            auto it = postconditionMap->find(dest);
            if (it != postconditionMap->end()) {
                it->second.erase(finalChunk);
                if (it->second.empty()) {
                    postconditionMap->erase(it);
                }
            }
        }
        
        // AllToAll-only: Remove ALL scheduled postconditions for this chunk
        // because chunk state has changed (either reached final dest or intermediate node)
        // This allows re-scheduling for next hop in multi-hop routes
        if (collectiveType_ == CollectiveType::ALL_TO_ALL) {
            for (const auto& finalDest : postconditions) {
                scheduledAllToAllPostconditions_.erase({finalChunk, finalDest});
            }
        }
        
        DebugLog(std::cout << "Chunk " << finalChunk << " arrived at GPU" << dest << std::endl);
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

int Synthesizer::linkChunkMatching_(const ChunkID chunk, const NpuID dest) noexcept {
    // Candidate selection: prefer earlier lower distance (only priority)
    auto minDistance = std::numeric_limits<Time>::max();
    std::vector<NpuID> candidates;

    // AllGather optimization: pre-filter sources to direct neighbors only
    const auto& directNeighbors = (collectiveType_ == CollectiveType::ALL_GATHER) 
        ? ten_->getDirectDeviceNeighbors(dest) 
        : std::unordered_set<NpuID>();
    
    // Iterate through all devices to find sources with the chunk
    for (int src = 0; src < npusCount; ++src) {
        if (src == dest) continue;  // skip self
        if (!chunkMap_[chunk][src]) {
            continue;  // source does not have the chunk
        }

        // AllGather: only consider direct neighbors (single-hop without crossing devices)
        // EXPERIMENT: Temporarily disable direct neighbor restriction to test multi-hop routing
        if (collectiveType_ == CollectiveType::ALL_GATHER) {
            if (directNeighbors.count(src) == 0) {
                // Allow multi-hop routing for sparse topologies
                // continue;  // not a direct neighbor
            }
        }

        const auto hopCount = ten_->getRoutePhysicalHops(src, dest);
        if (hopCount < 0) {
            continue;  // invalid route
        }

        const auto distance = ten_->getDistance(src, dest);

        // Priority: Earlier distance (lower distance is better)
        if (distance < minDistance) {
            // Found a source with lower distance - clear and start new candidate list
            minDistance = distance;
            candidates.clear();
            candidates.push_back(src);
        } else if (isEqual(distance, minDistance)) {
            // Same distance - add to candidates
            candidates.push_back(src);
        }
    }

    if (candidates.empty()) {
        return -1;
    }

    // Shuffle equal-priority candidates
    std::shuffle(candidates.begin(), candidates.end(), randomEngine);

    // Try each candidate source in shuffled order
    for (const auto selectedSrc : candidates) {
        bool success = false;
        
        if (collectiveType_ == CollectiveType::ALL_GATHER) {
            // AllGather: use direct precomputed routes (without passing through intermediate devices)
            success = tryAllGatherRouting_(chunk, selectedSrc, dest);
        } else if (collectiveType_ == CollectiveType::ALL_TO_ALL) {
            // AllToAll: use greedy routing (multi-hop with intermediate nodes)
            success = tryAllToAllRouting_(chunk, selectedSrc, dest);
        }

        if (success) {
            return selectedSrc;
        }
    }

    return -1;  // No candidate could reserve a route
}

bool Synthesizer::isEqual(const Time lhs, const Time rhs) noexcept {
    constexpr Time epsilon = 1e-9;
    return std::abs(lhs - rhs) < epsilon;
}

bool Synthesizer::tryAllGatherRouting_(const ChunkID chunk, const NpuID selectedSrc, const NpuID dest) noexcept {
    // Check if precomputed route is available
    if (!ten_->canReserveRoute(selectedSrc, dest)) {
        return false;  // Route not available
    }
    
    // Reserve and schedule the direct transfer
    const auto transferArrivalTime = currentTime_ + ten_->getDistance(selectedSrc, dest);
    ten_->transferChunk(selectedSrc, dest, chunk, transferArrivalTime);
    eventQueue_.schedule(transferArrivalTime);
    // Chunk arrival will be marked in expandTenTimestep_() when the event fires
    
    return true;
}

bool Synthesizer::tryAllToAllRouting_(const ChunkID chunk, const NpuID selectedSrc, const NpuID dest) noexcept {
    // Find best available next hop using greedy algorithm with route availability check
    const int greedyNextHop = ten_->findNextHopGreedy(selectedSrc, dest, currentTime_);
    
    if (greedyNextHop < 0) {
        return false;  // No progress possible or all routes busy
    }
    
    // Case 1: Direct hop to destination
    if (greedyNextHop == dest) {
        const auto directArrival = currentTime_ + ten_->getDistance(selectedSrc, dest);
        ten_->transferChunk(selectedSrc, dest, chunk, directArrival);
        eventQueue_.schedule(directArrival);
        // Chunk arrival will be marked in expandTenTimestep_() when the event fires
        return true;
    }
    
    // Case 2: Intermediate hop (greedy routing)
    const auto greedyArrivalTime = currentTime_ + ten_->getDistance(selectedSrc, greedyNextHop);
    ten_->transferChunkPartial(selectedSrc, greedyNextHop, chunk, greedyArrivalTime);
    eventQueue_.schedule(greedyArrivalTime);
    // Chunk arrival will be marked in expandTenTimestep_() when the event fires
    
    DebugLog(
        std::cout << "  [AllToAll] Partial (greedy-only): Chunk " << chunk
                  << " GPU" << selectedSrc << " -> GPU" << greedyNextHop
                  << " (final dest: GPU" << dest << ")" << std::endl;
    );
    
    // Dynamically adjust priority since chunk state changed (moved to intermediate node)
    adjustPostconditionPriority_(chunk, dest);
    
    return true;
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
    int totalRounds,
    const std::atomic<bool>& interruptFlag) noexcept {
    
    MultiRoundResult result;
    result.bestSynthesizer = nullptr;
    result.bestCollectiveTime = std::numeric_limits<Time>::max();
    result.bestRound = 0;
    result.totalRounds = 0;
    result.failedRounds = 0;
    result.totalSynthesisTime = 0.0;
    result.interrupted = false;
    
    for (int round = 1; round <= totalRounds; ++round) {
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
        
        // Check if synthesis failed (collectiveTime is -1)
        if (collectiveTime < 0) {
            result.failedRounds++;
            // Skip this failed round, don't update best result
            continue;
        }
        
        // Check if this is the best result
        if (collectiveTime < result.bestCollectiveTime) {
            result.bestCollectiveTime = collectiveTime;
            result.bestRound = result.totalRounds;
            result.bestSynthesizer = std::move(synthesizer);
        }
    }
    
    return result;
}

