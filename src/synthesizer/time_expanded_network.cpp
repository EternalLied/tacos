/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <cassert>
#include <memory>
#include <limits>
#include <random>
#include <queue>
#include <map>
#include <algorithm>
#include <tacos/synthesizer/time_expanded_network.h>

using namespace tacos;

TimeExpandedNetwork::TimeExpandedNetwork(const Topology& topology,
                                         const ChunkSize chunkSize) noexcept
    : topology_(topology) {
    assert(chunkSize > 0);
    npusCount_ = topology_.npusCount();
    totalNodes_ = topology_.totalNodes();

    // physical graph resources (will be properly initialized in computeEdgeTimes_)
    edgeCount_.assign(totalNodes_, std::vector<int>(totalNodes_, 0));
    edgeAccumulatedBusyTime_.assign(totalNodes_, std::vector<Time>(totalNodes_, 0));

    // per-switch caps & calendars
    swInCap_.assign(topology_.switchesCount(), 0);
    swOutCap_.assign(topology_.switchesCount(), 0);
    swAllowCopy_.assign(topology_.switchesCount(), 0);
    for (int sid = 0; sid < topology_.switchesCount(); ++sid) {
        swInCap_[sid] = topology_.switchInCapResolved(sid);
        swOutCap_[sid] = topology_.switchOutCapResolved(sid);
        swAllowCopy_[sid] = topology_.switchAt(sid).allowCopy ? 1 : 0;
    }
    swInUse_.assign(topology_.switchesCount(), {});
    swOutUse_.assign(topology_.switchesCount(), {});
    swInUseInsertCount_.assign(topology_.switchesCount(), 0);
    swOutUseInsertCount_.assign(topology_.switchesCount(), 0);

    // Adaptive cleanup threshold based on topology scale (linear growth)
    // Formula: max(200, npusCount_ * 20)
    // Analysis for switch_clique with N GPUs and K≤10 blocks:
    // - Theoretical max: N × (N-1) × K × 2 ≈ 20N(N-1)
    // - With concurrency and cleanup: peak ≈ 2N² to 6N²
    // - Empirical peaks (K≤10): 8→112-336, 16→480-1440, 32−2k-6k, 64−8k-24k, 128→33k-98k
    // - Linear threshold N×20: triggers cleanup every N×20 insertions per switch direction
    // - Cleanup based on insertion count (not vec.size) ensures memory is always bounded
    swCleanupThreshold_ = std::max(size_t(200), size_t(npusCount_ * 20));

    // per-edge Δ & GPU->GPU最短路
    computeEdgeTimes_(chunkSize);
    computeRoutes_(chunkSize);  // Also builds directDeviceNeighbors_ and distanceMatrix_

    // Initialize round-used edges tracker
    roundUsedEdges_.assign(totalNodes_, std::vector<char>(totalNodes_, 0));
}



int TimeExpandedNetwork::routeHopCount(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    
    if (src == dest) return 0;
    
    const auto& route = routes_[src][dest];
    if (route.nodes.empty() || route.nodes.size() < 2) {
        return -1; // no valid route
    }
    
    // Return logical hop count (Cut-Through switches reduce hop count)
    return route.logicalHops;
}

const std::unordered_set<TimeExpandedNetwork::NpuID>& 
TimeExpandedNetwork::getDirectDeviceNeighbors(const NpuID device) const noexcept {
    assert(0 <= device && device < npusCount_);
    return directDeviceNeighbors_[device];
}

void TimeExpandedNetwork::timestep(const Time time) noexcept {
    assert(time > currentTime_);
    // update the current timestep
    currentTime_ = time;
    // Note: Physical layer (edgeBusyUntil_) is checked dynamically in canReserveRoute_
}



bool TimeExpandedNetwork::canReserveRoute(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    
    if (src < 0 || src >= static_cast<int>(routes_.size())) return false;
    if (dest < 0 || dest >= static_cast<int>(routes_[src].size())) return false;
    
    const auto& route = routes_[src][dest];
    if (route.nodes.empty()) return false;
    
    return canReserveRoute_(route, currentTime_);
}

void TimeExpandedNetwork::clearRoundUsedEdges() noexcept {
    // Clear the round-used edges tracker at the start of each timestep matching
    for (auto& row : roundUsedEdges_) {
        std::fill(row.begin(), row.end(), 0);
    }
}

void TimeExpandedNetwork::transferChunk(const NpuID src,
                                        const NpuID dest,
                                        const ChunkID chunk,
                                        const Time time) noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(chunk >= 0);
    assert(time >= currentTime_);

    // Reserve physical route (checks should have been done via canReserveRoute)
    const auto& r = routes_[src][dest];
    reserveRoute_(r, currentTime_);
    
    // Record chunk arrival event with source information
    pendingArrivals_.emplace(time, Arrival{chunk, src, dest});
}



std::vector<int> TimeExpandedNetwork::getRoutePath(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    
    if (src >= routes_.size() || dest >= routes_[src].size()) {
        return {};
    }
    
    return routes_[src][dest].nodes;
}

int TimeExpandedNetwork::getRouteHops(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    
    if (src >= routes_.size() || dest >= routes_[src].size()) {
        return std::numeric_limits<int>::max();
    }
    
    const auto& route = routes_[src][dest];
    if (route.nodes.empty() || route.nodes.size() < 2) {
        return std::numeric_limits<int>::max();
    }
    
    // Return logical hop count (Cut-Through switches reduce hop count)
    return route.logicalHops;
}

int TimeExpandedNetwork::getRoutePhysicalHops(const NpuID src, const NpuID dest) const noexcept {
    if (src < 0 || src >= npusCount_ || dest < 0 || dest >= npusCount_) {
        return std::numeric_limits<int>::max();
    }
    
    const auto& route = routes_[src][dest];
    if (route.nodes.empty() || route.nodes.size() < 2) {
        return std::numeric_limits<int>::max();
    }
    
    // Return pre-cached physical hops (computed during route construction)
    return route.physicalHops;
}

double TimeExpandedNetwork::getLinkUtilization(const int src, const int dest, const Time totalTime) const noexcept {
    if (totalTime <= 0) {
        return 0.0;
    }
    
    if (src < 0 || src >= static_cast<int>(edgeAccumulatedBusyTime_.size()) || 
        dest < 0 || dest >= static_cast<int>(edgeAccumulatedBusyTime_[src].size())) {
        return 0.0;
    }
    
    // Use accumulated busy time for physical edge utilization
    const Time busyTime = edgeAccumulatedBusyTime_[src][dest];
    return static_cast<double>(busyTime) / static_cast<double>(totalTime);
}

void TimeExpandedNetwork::transferChunkPartial(const NpuID src, 
                                               const int intermediate, 
                                               const ChunkID chunk,
                                               const Time time) noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= intermediate && intermediate < npusCount_);
    assert(chunk >= 0);
    assert(time >= currentTime_);
    
    // Use the precomputed route from src to intermediate
    if (src >= static_cast<int>(routes_.size()) || 
        intermediate >= static_cast<int>(routes_[src].size())) {
        return; // Invalid route indices
    }
    
    const auto& partialRoute = routes_[src][intermediate];
    if (partialRoute.nodes.empty()) {
        return; // No route available
    }
    
    // Reserve the partial route's physical resources
    reserveRoute_(partialRoute, currentTime_);
    
    // Record chunk arrival at intermediate node with source information
    pendingArrivals_.emplace(time, Arrival{chunk, src, intermediate});
}

std::vector<std::tuple<TimeExpandedNetwork::ChunkID, TimeExpandedNetwork::NpuID, TimeExpandedNetwork::NpuID>> 
TimeExpandedNetwork::getArrivalsAt(Time time) noexcept {
    std::vector<std::tuple<ChunkID, NpuID, NpuID>> arrivals;
    
    // Find all arrivals at this time
    auto range = pendingArrivals_.equal_range(time);
    for (auto it = range.first; it != range.second; ++it) {
        const auto& arrival = it->second;
        arrivals.push_back(std::make_tuple(arrival.chunk, arrival.src, arrival.dest));
    }
    
    // Remove processed arrivals
    pendingArrivals_.erase(range.first, range.second);
    
    return arrivals;
}

// ===== helpers =====
void TimeExpandedNetwork::computeEdgeTimes_(const ChunkSize chunkSize) noexcept {
    // Initialize edge count matrix
    edgeCount_.assign(totalNodes_, std::vector<int>(totalNodes_, 0));
    
    // First pass: count parallel links between each (u,v) pair
    for (const auto& e : topology_.physLinks()) {
        edgeCount_[e.src][e.dst]++;
    }
    
    // Resize edge data structures based on parallel link counts
    edgeBusyUntil_.assign(totalNodes_, std::vector<std::vector<Time>>(totalNodes_));
    edgeDelta_.assign(totalNodes_, std::vector<std::vector<Time>>(totalNodes_));
    
    for (int u = 0; u < totalNodes_; ++u) {
        for (int v = 0; v < totalNodes_; ++v) {
            if (edgeCount_[u][v] > 0) {
                edgeBusyUntil_[u][v].resize(edgeCount_[u][v], -1);
                edgeDelta_[u][v].resize(edgeCount_[u][v], -1);
            }
        }
    }
    
    // Second pass: fill edge parameters for each parallel link
    std::vector<std::vector<int>> linkIndex(totalNodes_, std::vector<int>(totalNodes_, 0));
    
    for (const auto& e : topology_.physLinks()) {
        const double bwGiB = std::max(1e-12, (double)e.bw);
        const double beta_us = (double)chunkSize / (bwGiB * 1024.0 * 1024.0 * 1024.0) * 1e6;
        const double delta = e.alpha + beta_us;
        
        int idx = linkIndex[e.src][e.dst]++;
        edgeDelta_[e.src][e.dst][idx] = (Time)delta;
        edgeBusyUntil_[e.src][e.dst][idx] = -1;
    }
}

void TimeExpandedNetwork::computeRoutes_(const ChunkSize chunkSize) noexcept {
    // Initialize distanceMatrix_ here to avoid redundant computation in computeDistanceTable_
    distanceMatrix_.assign(npusCount_, std::vector<Time>(npusCount_, std::numeric_limits<Time>::max()));
    
    // Dijkstra per source GPU
    const int N = totalNodes_;
    routes_.assign(npusCount_, std::vector<Route>(npusCount_));
    // Build CUT_THROUGH-aware adjacency: for each u→SW→w path where SW is CUT_THROUGH,
    // add virtual edge u→w with pipelined time = max(d1, d2)
    // For parallel links, use the first link's delta (all parallel links have same parameters)
    std::vector<std::vector<std::pair<int, Time>>> adj(N);
    for (int u = 0; u < N; ++u) {
        for (int v = 0; v < N; ++v) if (edgeCount_[u][v] > 0) {
            int sid = topology_.switchIdFromNode(v);
            if (sid >= 0 && topology_.switchAt(sid).forwardingMode == SwitchForwardingMode::CUT_THROUGH) {
                // v is CUT_THROUGH switch: add virtual edges u→w for all neighbors w of v
                for (int w = 0; w < N; ++w) if (edgeCount_[v][w] > 0 && w != u) {
                    Time pipel = std::max(edgeDelta_[u][v][0], edgeDelta_[v][w][0]);
                    adj[u].push_back({w, pipel});
                }
            } else {
                // Not a CUT_THROUGH switch or regular node: add physical edge
                adj[u].push_back({v, edgeDelta_[u][v][0]});
            }
        }
    }
    
    // Helper to get alpha and beta for an edge
    auto getEdgeParams = [&](int u, int v) -> std::pair<double, double> {
        const auto& physLinks = topology_.physLinks();
        for (const auto& e : physLinks) {
            if (e.src == u && e.dst == v) {
                const double alpha = e.alpha;
                const double bwGiB = std::max(1e-12, (double)e.bw);
                const double beta = (double)chunkSize / (bwGiB * 1024.0 * 1024.0 * 1024.0) * 1e6;
                return {alpha, beta};
            }
        }
        return {0, 0};
    };
    
    // Helper to compute route time and logical hops based on switch forwarding mode
    // Cut-Through switches act as transparent pipelined links
    auto computeRouteTime = [&](const std::vector<int>& path) -> std::pair<Time, int> {
        if (path.size() < 2) return {0, 0};
        
        Time totalTime = 0;
        int totalLogicalHops = 0;
        
        size_t i = 0;
        while (i < path.size() - 1) {
            int u = path[i];
            int v = path[i + 1];
            
            // Check if v is a switch and path continues
            int switchId = topology_.switchIdFromNode(v);
            
            if (switchId >= 0 && i + 2 < path.size()) {
                // v is a switch, check its forwarding mode
                int w = path[i + 2]; // next node after switch
                const auto& sw = topology_.switchAt(switchId);
                
                if (sw.forwardingMode == SwitchForwardingMode::CUT_THROUGH) {
                    // Cut-through switch: acts as transparent pipelined link
                    // u → SW(CT) → w treated as single logical hop
                    // Time = max(α₁, α₂) + max(β₁, β₂)
                    auto [alpha1, beta1] = getEdgeParams(u, v);
                    auto [alpha2, beta2] = getEdgeParams(v, w);
                    
                    double maxAlpha = std::max(alpha1, alpha2);
                    double maxBeta = std::max(beta1, beta2);
                    
                    totalTime += (Time)(maxAlpha + maxBeta);
                    totalLogicalHops += 1; // Cut-through = 1 logical hop
                    
                    i += 2; // Skip both edges (u→v and v→w)
                } else {
                    // Store-and-forward switch: cumulative transmission
                    // Process first edge (u → v)
                    totalTime += edgeDelta_[u][v][0];
                    totalLogicalHops += 1;
                    i += 1;
                }
            } else {
                // Not a switch or last edge: cumulative
                totalTime += edgeDelta_[u][v][0];
                totalLogicalHops += 1;
                i += 1;
            }
        }
        
        return {totalTime, totalLogicalHops};
    };
    
    auto reconstruct = [&](int s, int t, const std::vector<int>& prev) {
        Route r;
        if (prev[t] < 0) return r;
        
        // Build Dijkstra path
        std::vector<int> dijkstraPath;
        for (int x = t; x != -1; x = prev[x]) dijkstraPath.push_back(x);
        std::reverse(dijkstraPath.begin(), dijkstraPath.end());
        
        // Expand virtual CUT_THROUGH edges to physical path
        std::vector<int> physicalPath;
        for (size_t i = 0; i < dijkstraPath.size(); ++i) {
            int u = dijkstraPath[i];
            physicalPath.push_back(u);
            
            if (i + 1 < dijkstraPath.size()) {
                int w = dijkstraPath[i + 1];
                // If no direct edge u→w, insert CUT_THROUGH switch
                if (edgeCount_[u][w] == 0) {
                    for (int sw = npusCount_; sw < totalNodes_; ++sw) {
                        if (edgeCount_[u][sw] > 0 && edgeCount_[sw][w] > 0) {
                            int sid = topology_.switchIdFromNode(sw);
                            if (sid >= 0 && topology_.switchAt(sid).forwardingMode == SwitchForwardingMode::CUT_THROUGH) {
                                physicalPath.push_back(sw);
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        r.nodes = physicalPath;
        auto [totalTime, logicalHops] = computeRouteTime(physicalPath);
        r.logicalHops = logicalHops;
        r.physicalHops = static_cast<int>(physicalPath.size()) - 1;
        
        for (size_t i = 1; i < physicalPath.size(); ++i) {
            r.deltas.push_back(edgeDelta_[physicalPath[i-1]][physicalPath[i]][0]);
        }
        
        return r;
    };
    
    // Build directDeviceNeighbors_ first (needed to decide which routes to store)
    directDeviceNeighbors_.assign(npusCount_, std::unordered_set<NpuID>());
    for (int u = 0; u < npusCount_; ++u) {
        for (int v = 0; v < npusCount_; ++v) {
            if (u == v) continue;
            
            bool isDirectNeighbor = false;
            
            // Case 1: Direct device-to-device edge
            if (edgeCount_[u][v] > 0) {
                isDirectNeighbor = true;
            }
            // Case 2: Path through switches only (no intermediate devices)
            else {
                // BFS to check if we can reach v from u without crossing devices
                std::queue<int> q;
                std::vector<bool> visited(totalNodes_, false);
                q.push(u);
                visited[u] = true;
                
                while (!q.empty() && !isDirectNeighbor) {
                    int curr = q.front();
                    q.pop();
                    
                    for (int next = 0; next < totalNodes_; ++next) {
                        if (edgeCount_[curr][next] == 0 || visited[next]) continue;
                        
                        if (next == v) {
                            isDirectNeighbor = true;
                            break;
                        }
                        
                        // Only continue through switches
                        if (next >= npusCount_) {  // next is a switch
                            visited[next] = true;
                            q.push(next);
                        }
                    }
                }
            }
            
            if (isDirectNeighbor) {
                directDeviceNeighbors_[u].insert(v);
            }
        }
    }
    
    // Now run Dijkstra once per source to populate both distanceMatrix_ and routes_
    // Random number generator for path selection (to explore different equal-cost paths)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    for (int sGPU = 0; sGPU < npusCount_; ++sGPU) {
        const int s = sGPU; // deviceNode == GPU id
        std::vector<Time> dist(N, std::numeric_limits<Time>::max()/4);
        std::vector<int>  prev(N, -1);
        using QN = std::pair<Time,int>;
        std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;
        dist[s] = 0; pq.push({0,s});
        
        // Epsilon for floating-point comparison
        constexpr Time epsilon = 1e-9;
        
        while(!pq.empty()) {
            auto [du,u] = pq.top(); pq.pop();
            if (du > dist[u] + epsilon) continue;  // Use epsilon comparison
            
            for (auto [v,w] : adj[u]) {
                Time newDist = du + w;
                
                if (dist[v] > newDist + epsilon) {
                    // Found strictly better path
                    dist[v] = newDist; 
                    prev[v] = u; 
                    pq.push({dist[v], v});
                } else if (std::abs(dist[v] - newDist) <= epsilon) {
                    // Found equal-cost path - randomly decide whether to replace
                    // This allows exploring different equal-cost paths across multiple runs
                    if (dis(gen) < 0.5) {
                        prev[v] = u;  // Replace with new path with 50% probability
                    }
                    // Note: We don't push to pq again since distance didn't improve
                }
            }
        }
        
        // Populate distanceMatrix_ for all reachable GPUs
        // Dijkstra distances are already CUT_THROUGH-aware from adjacency list
        for (int tGPU = 0; tGPU < npusCount_; ++tGPU) {
            if (sGPU != tGPU && topology_.connected(sGPU, tGPU)) {
                distanceMatrix_[sGPU][tGPU] = dist[tGPU];
            }
        }
        
        // Reconstruct routes only for direct neighbors
        for (int tGPU : directDeviceNeighbors_[sGPU]) {
            routes_[sGPU][tGPU] = reconstruct(s, tGPU, prev);
        }
    }
}

bool TimeExpandedNetwork::swCapOkAt_(const int sid, const Time s, const Time e, const bool isIn) const noexcept {
    const auto& vec = isIn ? swInUse_[sid] : swOutUse_[sid];
    const int cap = isIn ? swInCap_[sid] : swOutCap_[sid];
    constexpr Time epsilon = 1e-9;
    
    int overlap = 0;
    for (const auto& itv : vec) {
        bool noOverlap = (e < itv.s + epsilon) || (s > itv.e - epsilon);
        if (!noOverlap) {
            ++overlap;
            if (overlap >= cap) return false;
        }
    }
    return true;
}

void TimeExpandedNetwork::swReserve_(const int sid, const Time s, const Time e, const bool isIn) noexcept {
    auto& vec = isIn ? swInUse_[sid] : swOutUse_[sid];
    
    // Simply append the new interval
    vec.push_back({s, e});
    
    // Periodically clean up expired intervals based on insertion count
    // This ensures cleanup happens regardless of the ratio of expired/active intervals
    // and guarantees memory is bounded by swCleanupThreshold_ insertions worth of data
    size_t& insertCount = isIn ? swInUseInsertCount_[sid] : swOutUseInsertCount_[sid];
    if (++insertCount >= swCleanupThreshold_) {
        insertCount = 0;  // Reset counter after cleanup
        
        // Remove all expired intervals (those that ended before current time)
        auto it = std::remove_if(vec.begin(), vec.end(), 
            [this](const Interval& itv) { return itv.e < this->currentTime_; });
        vec.erase(it, vec.end());
    }
}

bool TimeExpandedNetwork::canReserveRoute_(const Route& r, const Time t0) const noexcept {
    if (r.nodes.size() < 2) return false;
    
    Time t = t0;
    size_t i = 1;
    
    while (i < r.nodes.size()) {
        int u = r.nodes[i-1], v = r.nodes[i];
        const Time d = r.deltas[i-1];
        
        // Check for CUT_THROUGH switch
        int switchId = topology_.switchIdFromNode(v);
        bool isCutThrough = (switchId >= 0 && i + 1 < r.nodes.size() && 
                             topology_.switchAt(switchId).forwardingMode == SwitchForwardingMode::CUT_THROUGH);
        
        if (isCutThrough) {
            int w = r.nodes[i + 1];
            const Time d2 = r.deltas[i];
            
            // Check if at least one parallel link pair is free for both edges
            if (edgeCount_[u][v] == 0 || edgeCount_[v][w] == 0) return false;
            
            bool foundFreePair = false;
            for (int idx1 = 0; idx1 < edgeCount_[u][v]; ++idx1) {
                if (edgeBusyUntil_[u][v][idx1] <= t) {
                    for (int idx2 = 0; idx2 < edgeCount_[v][w]; ++idx2) {
                        if (edgeBusyUntil_[v][w][idx2] <= t) {
                            foundFreePair = true;
                            break;
                        }
                    }
                    if (foundFreePair) break;
                }
            }
            if (!foundFreePair) return false;
            
            // Switch capacity for pipelined transmission
            Time maxDuration = std::max(d, d2);
            if (!swCapOkAt_(switchId, t, t + maxDuration, true)) return false;
            if (!swCapOkAt_(switchId, t, t + maxDuration, false)) return false;
            
            t += maxDuration;
            i += 2;
        } else {
            // Standard edge: check if any parallel link is free
            if (edgeCount_[u][v] == 0) return false;
            
            bool hasFreeLink = false;
            for (int idx = 0; idx < edgeCount_[u][v]; ++idx) {
                if (edgeBusyUntil_[u][v][idx] <= t) {
                    hasFreeLink = true;
                    break;
                }
            }
            if (!hasFreeLink) return false;
            
            const int sid_u = topology_.switchIdFromNode(u);
            const int sid_v = topology_.switchIdFromNode(v);
            if (sid_u >= 0 && !swCapOkAt_(sid_u, t, t + d, false)) return false;
            if (sid_v >= 0 && !swCapOkAt_(sid_v, t, t + d, true)) return false;
            
            t += d;
            i += 1;
        }
    }
    return true;
}

void TimeExpandedNetwork::reserveRoute_(const Route& r, const Time t0) noexcept {
    if (r.nodes.size() < 2) return;
    
    Time t = t0;
    size_t i = 1;
    
    while (i < r.nodes.size()) {
        int u = r.nodes[i-1], v = r.nodes[i];
        const Time d = r.deltas[i-1];
        
        // Check for CUT_THROUGH switch
        int switchId = topology_.switchIdFromNode(v);
        bool isCutThrough = (switchId >= 0 && i + 1 < r.nodes.size() && 
                             topology_.switchAt(switchId).forwardingMode == SwitchForwardingMode::CUT_THROUGH);
        
        if (isCutThrough) {
            int w = r.nodes[i + 1];
            const Time d2 = r.deltas[i];
            
            // Find the earliest available parallel link pair
            int bestIdx1 = -1, bestIdx2 = -1;
            for (int idx1 = 0; idx1 < edgeCount_[u][v]; ++idx1) {
                if (edgeBusyUntil_[u][v][idx1] <= t) {
                    for (int idx2 = 0; idx2 < edgeCount_[v][w]; ++idx2) {
                        if (edgeBusyUntil_[v][w][idx2] <= t) {
                            bestIdx1 = idx1;
                            bestIdx2 = idx2;
                            break;
                        }
                    }
                    if (bestIdx1 >= 0) break;
                }
            }
            
            // Reserve the selected parallel links
            if (bestIdx1 >= 0 && bestIdx2 >= 0) {
                edgeBusyUntil_[u][v][bestIdx1] = t + d;
                edgeBusyUntil_[v][w][bestIdx2] = t + d2;
                edgeAccumulatedBusyTime_[u][v] += d;
                edgeAccumulatedBusyTime_[v][w] += d2;
            }
            
            Time maxDuration = std::max(d, d2);
            swReserve_(switchId, t, t + maxDuration, true);
            swReserve_(switchId, t, t + maxDuration, false);
            
            t += maxDuration;
            i += 2;
        } else {
            // Standard edge: find earliest available parallel link
            int bestIdx = -1;
            for (int idx = 0; idx < edgeCount_[u][v]; ++idx) {
                if (edgeBusyUntil_[u][v][idx] <= t) {
                    bestIdx = idx;
                    break;
                }
            }
            
            // Reserve the selected parallel link
            if (bestIdx >= 0) {
                edgeBusyUntil_[u][v][bestIdx] = t + d;
                edgeAccumulatedBusyTime_[u][v] += d;
            }
            
            const int sid_u = topology_.switchIdFromNode(u);
            const int sid_v = topology_.switchIdFromNode(v);
            if (sid_u >= 0) swReserve_(sid_u, t, t + d, false);
            if (sid_v >= 0) swReserve_(sid_v, t, t + d, true);
            
            t += d;
            i += 1;
        }
    }
}

// ============================================================================
// AllToAll Optimization: Dynamic Greedy Routing Implementation
// ============================================================================

int TimeExpandedNetwork::findNextHopGreedy(const NpuID src, const NpuID dest, const Time currentTime) const noexcept {
    if (src < 0 || src >= npusCount_ || dest < 0 || dest >= npusCount_) {
        return -1;
    }
    
    if (src == dest) {
        return dest;  // Already at destination
    }
    
    // Check if dest is a direct neighbor
    if (directDeviceNeighbors_[src].count(dest) > 0) {
        // Check if route is available before returning
        if (src < static_cast<int>(routes_.size()) && 
            dest < static_cast<int>(routes_[src].size())) {
            const auto& route = routes_[src][dest];
            if (!route.nodes.empty() && canReserveRoute_(route, currentTime)) {
                return dest;  // Direct route available
            }
        }
        // Direct route exists but is busy - will try other neighbors below
    }
    
    // Get current distance from src to dest
    Time currentDist = getDistance(src, dest);
    if (currentDist >= std::numeric_limits<Time>::max() / 4) {
        return -1;  // Unreachable
    }
    
    // Collect all neighbors grouped by distance (for multi-tier fallback)
    // Map: distance -> list of neighbors at that distance
    std::map<Time, std::vector<int>> neighborsByDist;
    
    const auto& neighbors = directDeviceNeighbors_[src];
    for (const auto neighbor : neighbors) {
        Time neighborDist = getDistance(neighbor, dest);
        if (neighborDist >= currentDist) {
            continue;  // Not making progress
        }
        neighborsByDist[neighborDist].push_back(neighbor);
    }
    
    if (neighborsByDist.empty()) {
        return -1;  // No neighbors make progress
    }
    
    // Try each distance tier from best to worst
    std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
    
    for (const auto& [tierDist, tierNeighbors] : neighborsByDist) {
        // Shuffle neighbors at same distance for randomness
        std::vector<int> shuffledNeighbors = tierNeighbors;
        std::shuffle(shuffledNeighbors.begin(), shuffledNeighbors.end(), rng);
        
        // Try each neighbor in this tier
        for (const auto neighbor : shuffledNeighbors) {
            // Check if route to this neighbor is available
            if (src < static_cast<int>(routes_.size()) && 
                neighbor < static_cast<int>(routes_[src].size())) {
                const auto& route = routes_[src][neighbor];
                if (!route.nodes.empty() && canReserveRoute_(route, currentTime)) {
                    return neighbor;  // Found available route at this distance tier
                }
            }
        }
        // All neighbors at this tier are busy, try next tier (suboptimal but better than nothing)
    }
    
    return -1;  // No available neighbor found
}

TimeExpandedNetwork::Time TimeExpandedNetwork::getDistance(const NpuID src, const NpuID dest) const noexcept {
    if (src < 0 || src >= npusCount_ || dest < 0 || dest >= npusCount_) {
        return std::numeric_limits<Time>::max();
    }
    
    if (src == dest) {
        return 0;
    }
    
    // O(1) lookup in fast matrix (instead of O(N) linear scan)
    if (src < static_cast<int>(distanceMatrix_.size()) && 
        dest < static_cast<int>(distanceMatrix_[src].size())) {
        return distanceMatrix_[src][dest];
    }
    
    return std::numeric_limits<Time>::max();  // Unreachable
}
