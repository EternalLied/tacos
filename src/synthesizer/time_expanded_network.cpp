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
#include <algorithm>
#include <tacos/synthesizer/time_expanded_network.h>

using namespace tacos;

TimeExpandedNetwork::TimeExpandedNetwork(const Topology& topology,
                                         const ChunkSize chunkSize) noexcept
    : topology_(topology) {
    assert(chunkSize > 0);
    npusCount_ = topology_.npusCount();
    totalNodes_ = topology_.totalNodes();

    // initialize TEN lists
    linkBusyUntil_ = decltype(linkBusyUntil_)(npusCount_, std::vector<Time>(npusCount_, -1));
    chunk_ = decltype(chunk_)(npusCount_, std::vector<ChunkID>(npusCount_, -1));
    available_ = decltype(available_)(npusCount_, std::vector<bool>(npusCount_, false));
    linkTransferTimes_ =
        decltype(linkTransferTimes_)(npusCount_, std::vector<Time>(npusCount_, -1));

    // physical graph resources
    edgeBusyUntil_.assign(totalNodes_, std::vector<Time>(totalNodes_, -1));
    edgeDelta_.assign(totalNodes_, std::vector<Time>(totalNodes_, -1));
    hasEdge_.assign(totalNodes_, std::vector<char>(totalNodes_, 0));
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

    // per-edge Δ & GPU->GPU最短路
    computeEdgeTimes_(chunkSize);
    computeRoutes_(chunkSize);
    
    // AllToAll optimization: compute distance table and direct neighbors
    computeDistanceTable_(chunkSize);
    computeDirectNeighbors_();

    // Initialize round-used edges tracker
    roundUsedEdges_.assign(totalNodes_, std::vector<char>(totalNodes_, 0));

    for (int u = 0; u < npusCount_; ++u)
      for (int v = 0; v < npusCount_; ++v)
        available_[u][v] = topology_.connected(u, v);
}

bool TimeExpandedNetwork::available(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);

    // return true if the link is available at the current timestep
    return available_[src][dest];
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

std::unordered_set<TimeExpandedNetwork::NpuID> TimeExpandedNetwork::backtrack(
    const NpuID dest) noexcept {
    assert(0 <= dest && dest < npusCount_);

    // list of available source NPUs
    auto sources = std::unordered_set<NpuID>();

    // filter the available sources from the topology backtracking + route feasibility
    for (const auto src : topology_.backtrack(dest)) {
        if (!available_[src][dest]) continue; // gpu-level busy
        const auto& r = routes_[src][dest];
        if (r.nodes.empty()) continue;
        if (canReserveRoute_(r, currentTime_)) {
            sources.insert(src);
        }
    }

    return sources;
}

void TimeExpandedNetwork::timestep(const Time time) noexcept {
    assert(time > currentTime_);

    // update the current timestep
    currentTime_ = time;

    // reset the availability of all links
    for (auto src = 0; src < npusCount_; ++src) {
        for (auto dest = 0; dest < npusCount_; ++dest) {
            // if link is still busy, keep it unavailable
            const auto busyUntil = linkBusyUntil_[src][dest];
            if (busyUntil > currentTime_) {
                available_[src][dest] = false;
                continue;
            }

            // otherwise, reset the link availability
            available_[src][dest] = topology_.connected(src, dest);
        }
    }
}

TimeExpandedNetwork::ChunkID TimeExpandedNetwork::chunk(const NpuID src,
                                                        const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);

    // return the chunk ID being transferred over the link
    return chunk_[src][dest];
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

bool TimeExpandedNetwork::routeHasSwitch(const NpuID src, const NpuID dest) const noexcept {
    if (src < 0 || src >= static_cast<int>(routes_.size())) return false;
    if (dest < 0 || dest >= static_cast<int>(routes_[src].size())) return false;
    const auto& r = routes_[src][dest];
    if (r.nodes.size() < 2) return false;
    for (size_t i = 1; i < r.nodes.size(); ++i) {
        const int u = r.nodes[i-1];
        const int v = r.nodes[i];
        const int sid_u = topology_.switchIdFromNode(u);
        const int sid_v = topology_.switchIdFromNode(v);
        if (sid_u >= 0 || sid_v >= 0) return true;
    }
    return false;
}

bool TimeExpandedNetwork::routeHasMixedEdges(const NpuID src, const NpuID dest) const noexcept {
    if (src < 0 || src >= static_cast<int>(routes_.size())) return false;
    if (dest < 0 || dest >= static_cast<int>(routes_[src].size())) return false;
    const auto& r = routes_[src][dest];
    if (r.nodes.size() < 2) return false;
    bool hasSwitchEdge = false;
    bool hasDirectEdge = false;
    for (size_t i = 1; i < r.nodes.size(); ++i) {
        const int u = r.nodes[i-1];
        const int v = r.nodes[i];
        const bool isSwitchEdge = (topology_.switchIdFromNode(u) >= 0) || (topology_.switchIdFromNode(v) >= 0);
        if (isSwitchEdge) hasSwitchEdge = true; else hasDirectEdge = true;
        if (hasSwitchEdge && hasDirectEdge) return true;
    }
    return false;
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

    // assert link is currently available and free
    assert(available_[src][dest]);
    assert(linkBusyUntil_[src][dest] < 0);

    // reserve physical route now (atomic multi-segment)
    const auto& r = routes_[src][dest];
    reserveRoute_(r, currentTime_);
    // mark GPU-level virtual link
    available_[src][dest] = false;
    chunk_[src][dest] = chunk;
    linkBusyUntil_[src][dest] = time;

}

void TimeExpandedNetwork::transferFinished(const NpuID src, const NpuID dest) noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);

    // reset the link busy time and chunk
    available_[src][dest] = true;
    linkBusyUntil_[src][dest] = -1;
    chunk_[src][dest] = -1;

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

int TimeExpandedNetwork::getFirstIntermediateDevice(const NpuID src, const NpuID dest) const noexcept {
    if (src < 0 || src >= npusCount_ || dest < 0 || dest >= npusCount_) {
        return -1;
    }
    
    const auto& route = routes_[src][dest];
    if (route.nodes.size() < 3) {
        // No intermediate nodes (direct connection or too short)
        return -1;
    }
    
    // Search for first intermediate device node (excluding src and dest)
    for (size_t i = 1; i < route.nodes.size() - 1; ++i) {
        int nodeIdx = route.nodes[i];
        // Check if this node is a device (not a switch)
        if (nodeIdx < npusCount_) {
            return nodeIdx;
        }
    }
    
    return -1; // No intermediate device found
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
    
    // Mark virtual link src->intermediate as busy
    available_[src][intermediate] = false;
    chunk_[src][intermediate] = chunk;
    linkBusyUntil_[src][intermediate] = time;
}

TimeExpandedNetwork::Time TimeExpandedNetwork::linkTransferTime(const NpuID src,
                                                                const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(topology_.connected(src, dest));

    const auto& linkTime = routes_[src][dest].total;
    // const auto linkTime = linkTransferTimes_[src][dest];
    assert(linkTime >= 0);

    return linkTime;
}

void TimeExpandedNetwork::computeLinkTimes_(const ChunkSize chunkSize) noexcept {
    assert(chunkSize > 0);

    // for all links
    for (auto src = 0; src < npusCount_; ++src) {
        for (auto dest = 0; dest < npusCount_; ++dest) {
            if (!topology_.connected(src, dest)) {
                continue;
            };

            // use alpha-beta model to calculate link transfer time
            const auto bandwidth = topology_.bandwidth(src, dest);
            const auto latency = topology_.latency(src, dest);
            const auto linkTime = alphaBetaModel_(bandwidth, latency, chunkSize);
            linkTransferTimes_[src][dest] = linkTime;
        }
    }
}

TimeExpandedNetwork::Time TimeExpandedNetwork::alphaBetaModel_(const Bandwidth bandwidth,
                                                               const Latency latency,
                                                               const ChunkSize chunkSize) noexcept {
    assert(bandwidth > 0);
    assert(latency >= 0);
    assert(chunkSize > 0);

    // convert bandwidth from GiB/sec to bytes/microseconds
    const auto bandwidthConverted = bandwidth * (1 << 30) / 1e6;  // bytes/microseconds

    // run alpha-beta model
    const auto alpha = latency;
    const auto beta = chunkSize / bandwidthConverted;

    return alpha + beta;
}

// ===== helpers =====
void TimeExpandedNetwork::computeEdgeTimes_(const ChunkSize chunkSize) noexcept {
    // init to no-edge
    for (int u = 0; u < totalNodes_; ++u)
      for (int v = 0; v < totalNodes_; ++v) {
        edgeBusyUntil_[u][v] = -1;
        edgeDelta_[u][v] = -1;
        hasEdge_[u][v] = 0;
      }
    // fill from topology.physLinks
    for (const auto& e : topology_.physLinks()) {
        const double bwGiB = std::max(1e-12, (double)e.bw);
        const double beta_us = (double)chunkSize / (bwGiB * 1024.0 * 1024.0 * 1024.0) * 1e6; // GiB/s -> μs
        const double delta = e.alpha + beta_us;
        edgeDelta_[e.src][e.dst] = (Time)delta;  // Keep precision, don't round
        hasEdge_[e.src][e.dst] = 1;
        edgeBusyUntil_[e.src][e.dst] = -1;
    }
}

void TimeExpandedNetwork::computeRoutes_(const ChunkSize chunkSize) noexcept {
    // Dijkstra per source GPU
    const int N = totalNodes_;
    routes_.assign(npusCount_, std::vector<Route>(npusCount_));
    // adjacency
    std::vector<std::vector<std::pair<int, Time>>> adj(N);
    for (int u = 0; u < N; ++u) {
        for (int v = 0; v < N; ++v) if (hasEdge_[u][v]) {
            adj[u].push_back({v, edgeDelta_[u][v]});
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
                    totalTime += edgeDelta_[u][v];
                    totalLogicalHops += 1;
                    i += 1;
                }
            } else {
                // Not a switch or last edge: cumulative
                totalTime += edgeDelta_[u][v];
                totalLogicalHops += 1;
                i += 1;
            }
        }
        
        return {totalTime, totalLogicalHops};
    };
    
    auto reconstruct = [&](int s, int t, const std::vector<int>& prev) {
        Route r;
        if (prev[t] < 0) return r;
        std::vector<int> path;
        for (int x = t; x != -1; x = prev[x]) path.push_back(x);
        std::reverse(path.begin(), path.end());
        r.nodes = path;
        
        // Compute total time and logical hops based on switch mode
        auto [totalTime, logicalHops] = computeRouteTime(path);
        r.total = totalTime;
        r.logicalHops = logicalHops;
        r.physicalHops = static_cast<int>(path.size()) - 1; // Cache physical hops
        
        // Still store per-edge deltas for reservation purposes
        for (size_t i = 1; i < path.size(); ++i) {
            int u = path[i-1], v = path[i];
            r.deltas.push_back(edgeDelta_[u][v]);
        }
        
        return r;
    };
    
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
        for (int tGPU = 0; tGPU < npusCount_; ++tGPU) {
            if (sGPU == tGPU) continue;
            if (!topology_.connected(sGPU, tGPU)) continue;
            const int t = tGPU;
            routes_[sGPU][tGPU] = reconstruct(s, t, prev);
            linkTransferTimes_[sGPU][tGPU] = routes_[sGPU][tGPU].total;
        }
    }
}

bool TimeExpandedNetwork::swCapOkAt_(const int sid, const Time s, const Time e, const bool isIn) const noexcept {
    const auto& vec = isIn ? swInUse_[sid] : swOutUse_[sid];
    const int cap = isIn ? swInCap_[sid] : swOutCap_[sid];
    // count overlaps in [s,e)
    int overlap = 0;
    for (const auto& itv : vec) {
        if (!(e <= itv.s || s >= itv.e)) { // overlap
            ++overlap;
            if (overlap >= cap) return false;
        }
    }
    return true;
}

void TimeExpandedNetwork::swReserve_(const int sid, const Time s, const Time e, const bool isIn) noexcept {
    auto& vec = isIn ? swInUse_[sid] : swOutUse_[sid];
    vec.push_back({s,e});
}

bool TimeExpandedNetwork::canReserveRoute_(const Route& r, const Time t0) const noexcept {
    if (r.nodes.size() < 2) return false;
    Time t = t0;
    for (size_t i = 1; i < r.nodes.size(); ++i) {
        int u = r.nodes[i-1], v = r.nodes[i];
        const Time d = r.deltas[i-1];
        // link must be free at t
        if (!hasEdge_[u][v]) return false;
        if (edgeBusyUntil_[u][v] > t) {
            // Edge is busy - route cannot be reserved
            return false;
        }
        // OPTIMIZATION: Removed roundUsedEdges check to allow edge reuse at different times
        // The edgeBusyUntil check above already prevents actual time conflicts
        // Removing this constraint increases parallelism significantly
        // if (roundUsedEdges_[u][v]) {
        //     return false;
        // }
        // switch caps at u/v if they are switches
        const int sid_u = topology_.switchIdFromNode(u);
        const int sid_v = topology_.switchIdFromNode(v);
        // entering v (if v is switch) consumes its IN; leaving u (if u is switch) consumes its OUT
        if (sid_u >= 0) { // leaving switch u
            if (!swCapOkAt_(sid_u, t, t + d, /*isIn=*/false)) return false;
        }
        if (sid_v >= 0) { // entering switch v
            if (!swCapOkAt_(sid_v, t, t + d, /*isIn=*/true)) return false;
        }
        t += d;
    }
    return true;
}

void TimeExpandedNetwork::reserveRoute_(const Route& r, const Time t0) noexcept {
    Time t = t0;
    for (size_t i = 1; i < r.nodes.size(); ++i) {
        int u = r.nodes[i-1], v = r.nodes[i];
        const Time d = r.deltas[i-1];
        // mark edge busy
        edgeBusyUntil_[u][v] = t + d;
        // Accumulate busy time for this physical edge
        edgeAccumulatedBusyTime_[u][v] += d;
        // OPTIMIZATION: No longer marking roundUsedEdges since we removed the constraint
        // roundUsedEdges_[u][v] = 1;
        // switch caps
        const int sid_u = topology_.switchIdFromNode(u);
        const int sid_v = topology_.switchIdFromNode(v);
        if (sid_u >= 0) swReserve_(sid_u, t, t + d, /*isIn=*/false);
        if (sid_v >= 0) swReserve_(sid_v, t, t + d, /*isIn=*/true);
        t += d;
    }
}

// ============================================================================
// AllToAll Optimization: Dynamic Greedy Routing Implementation
// ============================================================================

void TimeExpandedNetwork::computeDistanceTable_(const ChunkSize chunkSize) noexcept {
    // Initialize distance tables
    distanceTable_.assign(npusCount_, std::vector<std::pair<NpuID, Time>>());
    distanceMatrix_.assign(npusCount_, std::vector<Time>(npusCount_, std::numeric_limits<Time>::max()));
    directLinkTimes_.assign(npusCount_, std::vector<Time>(npusCount_, -1));
    
    // For each target NPU, run Dijkstra to get distances from all other NPUs
    for (int target = 0; target < npusCount_; ++target) {
        std::vector<Time> dist(totalNodes_, std::numeric_limits<Time>::max() / 4);
        using QN = std::pair<Time, int>;
        std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;
        
        dist[target] = 0;
        pq.push({0, target});
        
        while (!pq.empty()) {
            auto [d_u, u] = pq.top();
            pq.pop();
            if (d_u > dist[u]) continue;
            
            for (int v = 0; v < totalNodes_; ++v) {
                if (hasEdge_[u][v]) {
                    Time newDist = d_u + edgeDelta_[u][v];
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                        pq.push({newDist, v});
                    }
                }
            }
        }
        
        // Store distances from all NPUs to this target, sorted
        for (int src = 0; src < npusCount_; ++src) {
            if (src != target && dist[src] < std::numeric_limits<Time>::max() / 4) {
                distanceTable_[target].push_back({src, dist[src]});
                // Also populate fast lookup matrix for O(1) access
                distanceMatrix_[src][target] = dist[src];
            }
        }
        
        // Sort by distance (ascending)
        std::sort(distanceTable_[target].begin(), distanceTable_[target].end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
    }
    
    // Store direct link times between NPUs
    for (int u = 0; u < npusCount_; ++u) {
        for (int v = 0; v < npusCount_; ++v) {
            if (u != v && hasEdge_[u][v]) {
                directLinkTimes_[u][v] = edgeDelta_[u][v];
            }
        }
    }
}

void TimeExpandedNetwork::computeDirectNeighbors_() noexcept {
    // Initialize
    directDeviceNeighbors_.assign(npusCount_, std::unordered_set<NpuID>());
    
    // For each NPU, find direct device neighbors
    // Direct neighbor = connected by a path that doesn't cross another device
    for (int u = 0; u < npusCount_; ++u) {
        for (int v = 0; v < npusCount_; ++v) {
            if (u == v) continue;
            
            // Check if there's a path from u to v without crossing another device
            bool isDirectNeighbor = false;
            
            // Case 1: Direct device-to-device edge
            if (hasEdge_[u][v]) {
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
                        if (!hasEdge_[curr][next] || visited[next]) continue;
                        
                        if (next == v) {
                            // Reached target
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
}

int TimeExpandedNetwork::findNextHopGreedy(const NpuID src, const NpuID dest) const noexcept {
    if (src < 0 || src >= npusCount_ || dest < 0 || dest >= npusCount_) {
        return -1;
    }
    
    if (src == dest) {
        return dest;  // Already at destination - return dest to indicate success
    }
    
    // Check if dest is a direct neighbor - if so, return it directly
    if (directDeviceNeighbors_[src].count(dest) > 0) {
        return dest;  // Can go directly to destination
    }
    
    // Get current distance from src to dest
    Time currentDist = getDistance(src, dest);
    if (currentDist >= std::numeric_limits<Time>::max() / 4) {
        return -1;  // Unreachable
    }
    
    // Collect best neighbors (strictly closer) and pick one at random
    int bestNeighbor = -1;
    Time bestDist = currentDist;  // Must be strictly better
    std::vector<int> bestNeighbors;
    
    const auto& neighbors = directDeviceNeighbors_[src];
    for (const auto neighbor : neighbors) {
        Time neighborDist = getDistance(neighbor, dest);
        if (neighborDist >= currentDist) {
            continue;  // Not making progress
        }

        if (neighborDist < bestDist) {
            bestDist = neighborDist;
            bestNeighbors.clear();
            bestNeighbors.push_back(neighbor);
        } else if (neighborDist == bestDist) {
            bestNeighbors.push_back(neighbor);
        }
    }

    if (!bestNeighbors.empty()) {
        // Simple random choice among equally good neighbors
        std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
        std::uniform_int_distribution<size_t> dist(0, bestNeighbors.size() - 1);
        bestNeighbor = bestNeighbors[dist(rng)];
    }

    return bestNeighbor;
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
