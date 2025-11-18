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
#include <tacos/synthesizer/synthesizer.h>
#include "log.h"

using namespace tacos;

Synthesizer::Synthesizer() noexcept = default;

Synthesizer::Time Synthesizer::solve(const Topology& topology,
                                     const Collective& collective,
                                     ChunkSize chunkSize) noexcept {
    assert(chunkSize > 0);

    // initialize the synthesizer
    initialize_(topology, collective, chunkSize);

    // mark trivial initial case
    // that is, chunks in preconditions are already at their sources
    markPrecondition_();

    // then, repeat the link-chunk matching process
    while (!eventQueue_.empty()) {
        // get current event time
        currentTime_ = eventQueue_.pop();
        DebugLog(std::cout << "[TacosEvent]" << std::endl);
        DebugLog(std::cout << "At Time: " << currentTime_ << std::endl);

        // first, filter out unsatisfied postconditions
        // this is required when choosing the chunk replacement candidates
        // during the expansion of the TEN
        auto postconditionMap = filterPostcondition_();

        // then, expand the TEN
        // this method will also process and update the arrival of chunks
        // at the current timestep, and will change the unsatisfied postconditions
        // and return the number of replacements performed
        const auto [replacedCount, discardedCount] = expandTenTimestep_(&postconditionMap);
        DebugLog(std::cout << "Replaced: " << replacedCount << std::endl);
        DebugLog(std::cout << "Discarded: " << discardedCount << std::endl);

        // count total number of unsatisfied chunk requests across all destinations
        auto unsatisfiedCount = 0;
        for (const auto &kv : postconditionMap) unsatisfiedCount += static_cast<int>(kv.second.size());
        DebugLog(std::cout << "unsatisfied postconditions: " << unsatisfiedCount << std::endl);

        // after the expansion of the TEN, check if there are any unsatisfied postconditions
        auto postcondition = shufflePostcondition_(postconditionMap);

        if (postcondition.empty()) {
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
        // - If shortest path has no switch, require 1-hop only
        // - If path includes switches, prioritize fewer hops (no global cap)
        // - Reject routes mixing direct device edges with switch edges
        // All policy logic is implemented inside linkChunkMatching_

        // for all unsatisfied postconditions, run link-chunk matching
        for (const auto [chunk, dest] : postcondition) {
            // linkChunkMatching_ returns the selected source NPU, or -1 if failed
            const auto selectedSrc = linkChunkMatching_(chunk, dest);
            
            if (selectedSrc >= 0) {
                // Get the actual route path that was reserved
                auto path = ten_->getRoutePath(selectedSrc, dest);
                matchedRoutes.push_back({chunk, selectedSrc, dest, path});
                ++successfulMatchingCount;
            }
        }

        DebugLog(std::cout << "Scheduled: " << successfulMatchingCount << std::endl);
        
        // Print matched routes with hop count
        DebugLog(
            int totalHops = 0;
            for (const auto& [chunk, src, dest, path] : matchedRoutes) {
                int hops = path.empty() ? 0 : static_cast<int>(path.size()) - 1;
                totalHops += hops;
                std::cout << "  Chunk " << chunk << ": GPU" << src << " -> GPU" << dest 
                          << " [" << hops << " hop" << (hops != 1 ? "s" : "") << "]";
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
                double avgHops = static_cast<double>(totalHops) / matchedRoutes.size();
                std::cout << "  Average hops: " << std::fixed << std::setprecision(2) << avgHops << std::endl;
            }
        );
        DebugLog(std::cout << std::endl);
    }

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
    auto postcondition = std::vector<Condition>();

    // flatten the postcondition map into a vector of conditions
    for (const auto& [dest, chunks] : postconditionMap) {
        for (const auto chunk : chunks) {
            postcondition.emplace_back(chunk, dest);
        }
    }

    // Sort by minimum hop count (shortest path) to prioritize direct/short paths
    // This improves link utilization by avoiding long multi-hop paths when possible
    std::stable_sort(postcondition.begin(), postcondition.end(),
        [this](const Condition& a, const Condition& b) {
            const auto [chunkA, destA] = a;
            const auto [chunkB, destB] = b;
            
            // Find minimum hop count for condition A
            int minHopsA = std::numeric_limits<int>::max();
            auto sourcesA = ten_->backtrack(destA);
            for (const auto src : sourcesA) {
                if (chunkMap_[chunkA][src]) {
                    int hops = ten_->getRouteHops(src, destA);
                    minHopsA = std::min(minHopsA, hops);
                }
            }
            
            // Find minimum hop count for condition B
            int minHopsB = std::numeric_limits<int>::max();
            auto sourcesB = ten_->backtrack(destB);
            for (const auto src : sourcesB) {
                if (chunkMap_[chunkB][src]) {
                    int hops = ten_->getRouteHops(src, destB);
                    minHopsB = std::min(minHopsB, hops);
                }
            }
            
            return minHopsA < minHopsB;
        });
    
    // Shuffle conditions with the same hop count to add randomness
    // This prevents bias towards specific chunks/destinations with same hop count
    auto start = postcondition.begin();
    for (auto it = postcondition.begin(); it != postcondition.end(); ) {
        const auto [chunk, dest] = *it;
        
        // Find minimum hop count for current condition
        int currentHops = std::numeric_limits<int>::max();
        auto sources = ten_->backtrack(dest);
        for (const auto src : sources) {
            if (chunkMap_[chunk][src]) {
                int hops = ten_->getRouteHops(src, dest);
                currentHops = std::min(currentHops, hops);
                break; // Just need to know the minimum
            }
        }
        
        // Find end of same-hop-count group
        auto groupEnd = it;
        while (groupEnd != postcondition.end()) {
            const auto [chunkG, destG] = *groupEnd;
            int groupHops = std::numeric_limits<int>::max();
            auto sourcesG = ten_->backtrack(destG);
            for (const auto src : sourcesG) {
                if (chunkMap_[chunkG][src]) {
                    int hops = ten_->getRouteHops(src, destG);
                    groupHops = std::min(groupHops, hops);
                    break;
                }
            }
            if (groupHops != currentHops) break;
            ++groupEnd;
        }
        
        // Shuffle within this group
        std::shuffle(it, groupEnd, randomEngine);
        it = groupEnd;
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
            // and if so, check if we can replace this path with another chun
            if (chunkMap_[chunk][dest]) {
                // dest has already received this chunk
                // so check the replacement candidates
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

int Synthesizer::linkChunkMatching_(const ChunkID chunk, const NpuID dest) noexcept {
    // backtrack source NPUs
    auto sources = ten_->backtrack(dest);

    // filter candidate link-chunk matching
    // prioritize by: 1) minimum hop count, 2) earliest arrival time
    auto minHopCount = std::numeric_limits<int>::max();
    auto arrivalTime = std::numeric_limits<Time>::max();
    auto candidates = std::vector<NpuID>();

    // iterate over all source NPUs
    for (const auto src : sources) {
        // if src does not have the chunk, skip
        if (!chunkMap_[chunk][src]) {
            continue;
        }

        // get hop count and transfer time for this route
        const auto hopCount = ten_->routeHopCount(src, dest);
        const auto linkWeight = ten_->linkTransferTime(src, dest);
        const auto linkTime = currentTime_ + linkWeight;

        // skip invalid routes
        if (hopCount < 0) continue;

        // Policy enforcement (per-route):
        // 1) If topology has no switches, only schedule 1-hop routes
        if (!ten_->hasSwitches() && hopCount != 1) continue;

        // 2) If the route's shortest path does not traverse any switch, require 1-hop
        const bool routeThroughSwitch = ten_->routeHasSwitch(src, dest);
        if (!routeThroughSwitch && hopCount != 1) continue;

        // 3) Do not match routes that mix direct device edges with switch edges
        if (ten_->routeHasMixedEdges(src, dest)) continue;

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
