/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <memory>
#include <tacos/collective/collective.h>
#include <tacos/topology/topology.h>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <limits>

namespace tacos {

/// @brief Time-expanded network for synthesizing collective patterns.
class TimeExpandedNetwork {
  public:
    // data types
    using Time = Topology::Time;
    using NpuID = Topology::NpuID;
    using ChunkID = Collective::ChunkID;
    using ChunkSize = Collective::ChunkSize;
    using Bandwidth = Topology::Bandwidth;
    using Latency = Topology::Latency;

    /// @brief Construct the time-expanded network
    /// @param topology target network topology
    TimeExpandedNetwork(const Topology& topology, ChunkSize chunkSize) noexcept;

    /// @brief Check if a link is available at the current timestep
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return true if the TEN link is available, false otherwise
    [[nodiscard]] bool available(NpuID src, NpuID dest) const noexcept;

    /// @brief Retrieve the chunk transfer time between two NPUs
    /// @brief Get the number of hops in the route from src to dest
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return number of hops (edges) in the route, or -1 if no route exists
    [[nodiscard]] int routeHopCount(NpuID src, NpuID dest) const noexcept;

    /// @brief Occupy a link between two NPUs
    /// @details i.e., mark the link as unavailable for the current timestep.
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    void disable(NpuID src, NpuID dest) noexcept;

    /// @brief Reset to a new timestep
    /// @details This resets the availability of all links in the network
    /// @param nextTime next timestep to set
    void timestep(Time time) noexcept;

    /// @brief Get the chunk currently being trasferred over a link
    /// @details If the link is currently free, returns a negative number.
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return chunk ID being transferred over the link
    [[nodiscard]] ChunkID chunk(NpuID src, NpuID dest) const noexcept;

    /// @brief Mark a chunk as being transferred over a link
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @param chunk chunk ID being transferred over the link
    /// @param time time until which the link is busy
    void transferChunk(NpuID src, NpuID dest, ChunkID chunk, Time time) noexcept;

    /// @brief Mark a chunk transfer as finished over a link
    /// @details This resets the chunk information and link busy time.
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    void transferFinished(NpuID src, NpuID dest) noexcept;

    /// @brief Get the route (path) from src to dest
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return vector of node indices representing the path
    [[nodiscard]] std::vector<int> getRoutePath(NpuID src, NpuID dest) const noexcept;

    /// @brief Get the number of logical hops in the route from src to dest
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return number of logical hops (Cut-Through switches count as 1 hop)
    [[nodiscard]] int getRouteHops(NpuID src, NpuID dest) const noexcept;

    /// @brief Get the number of physical hops (edges) in the route from src to dest
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return number of physical hops (actual edges traversed)
    [[nodiscard]] int getRoutePhysicalHops(NpuID src, NpuID dest) const noexcept;

    /// @brief Calculate link utilization for a given link
    /// @param src source node ID
    /// @param dest destination node ID
    /// @param totalTime total collective time
    /// @return utilization ratio (0.0 to 1.0)
    [[nodiscard]] double getLinkUtilization(int src, int dest, Time totalTime) const noexcept;

    /// @brief Check if a route can be reserved at the current time
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return true if the route is available, false otherwise
    [[nodiscard]] bool canReserveRoute(NpuID src, NpuID dest) const noexcept;

    /// @brief Find best available next hop for AllToAll greedy routing
    /// @details Selects neighbor that is closer to dest and has available link.
    ///          Checks route availability and falls back to suboptimal routes if needed.
    /// @param src current source NPU ID
    /// @param dest final destination NPU ID
    /// @param currentTime current timestep for route availability check
    /// @return next hop NPU ID, or -1 if no progress can be made
    [[nodiscard]] int findNextHopGreedy(NpuID src, NpuID dest, Time currentTime) const noexcept;

    /// @brief Get distance from src to dest (precomputed transfer time)
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return transfer time, or max if unreachable
    [[nodiscard]] Time getDistance(NpuID src, NpuID dest) const noexcept;
    
    /// @brief Get the set of direct device neighbors for a given device
    /// @details A direct neighbor is reachable via a single-hop path without crossing other devices
    ///          (may go through switches but no intermediate devices)
    /// @param device device NPU ID
    /// @return set of direct device neighbor IDs
    [[nodiscard]] const std::unordered_set<NpuID>& getDirectDeviceNeighbors(NpuID device) const noexcept;

    /// @brief Transfer chunk using partial route (only to first intermediate device)
    /// @details For AllToAll multi-hop optimization: reserve only src->intermediate segment
    /// @param src source NPU ID
    /// @param intermediate intermediate device node
    /// @param chunk chunk ID being transferred
    /// @param time arrival time at intermediate node
    void transferChunkPartial(NpuID src, int intermediate, ChunkID chunk, Time time) noexcept;

    /// @brief Clear the current-round used edges set (call at start of each timestep matching)
    void clearRoundUsedEdges() noexcept;

  /// @brief Whether the underlying topology has any switches
  [[nodiscard]] bool hasSwitches() const noexcept { return topology_.switchesCount() > 0; }

  /// @brief Whether a node index is a switch node
  [[nodiscard]] bool nodeIsSwitch(int nodeIndex) const noexcept {
    return topology_.switchIdFromNode(nodeIndex) >= 0;
  }

  private:
    /// @brief current timestep
    Time currentTime_ = -1;

    /// @brief target network topology
    const Topology& topology_;

    /// @brief number of NPUs in the topology
    int npusCount_ = -1;
    int totalNodes_ = 0; // devices + switches

    /// @brief true if src-dest link is available at the current timestep
    std::vector<std::vector<bool>> available_;

    /// @brief time until which the link src-dest is busy
    /// @details if the link is free, the value is negative.
    std::vector<std::vector<Time>> linkBusyUntil_;

    /// @brief current chunk being transferred over the link src-dest
    /// @details if the link is free, the value is negative.
    std::vector<std::vector<ChunkID>> chunk_;

    // ===== Multi-switch physical resources =====
    struct EdgeKey { int u; int v; };
    // busy-until on physical directed edges (device/switch graph)
    std::vector<std::vector<Time>> edgeBusyUntil_; // sized [totalNodes_][totalNodes_], -1 if no edge
    std::vector<std::vector<Time>> edgeDelta_;     // per-edge α+β·n (μs); -1 if no edge
    std::vector<std::vector<char>> hasEdge_;       // quick check
    
    // Physical edge utilization tracking (for statistics)
    std::vector<std::vector<Time>> edgeAccumulatedBusyTime_; // total busy time for utilization stats

    // per-switch in/out concurrent usage calendars: intervals [start,end)
    struct Interval { Time s, e; };
    std::vector<std::vector<Interval>> swInUse_;   // [sid] list of intervals
    std::vector<std::vector<Interval>> swOutUse_;  // [sid] list of intervals
    std::vector<int> swInCap_;   // resolved caps
    std::vector<int> swOutCap_;
    std::vector<char> swAllowCopy_;
    size_t swCleanupThreshold_;  // adaptive threshold for interval cleanup
    
    // Track insertion count per switch direction for periodic cleanup
    std::vector<size_t> swInUseInsertCount_;   // [sid] insertions since last cleanup
    std::vector<size_t> swOutUseInsertCount_;  // [sid] insertions since last cleanup

    // precomputed shortest route for each GPU pair at current chunkSize
    struct Route {
      std::vector<int> nodes;     // node indices u->...->v
      std::vector<Time> deltas;   // per-edge Δ
      int logicalHops = 0;        // logical hop count (Cut-Through switches reduce hops)
      int physicalHops = 0;       // physical hop count (actual edges = nodes.size() - 1)
    };
    std::vector<std::vector<Route>> routes_; // [srcGPU][dstGPU]

    // Track edges used in current matching round to prevent path conflicts
    // within the same timestep (even if used at different times)
    std::vector<std::vector<char>> roundUsedEdges_; // [u][v] = 1 if edge used this round

    // === AllToAll Optimization: Dynamic Greedy Routing ===
    // Fast lookup matrix: distanceMatrix_[src][dest] = distance from src to dest (O(1) lookup)
    std::vector<std::vector<Time>> distanceMatrix_;
    
    // Direct device neighbors: NPUs that are directly connected without crossing another device
    std::vector<std::unordered_set<NpuID>> directDeviceNeighbors_;

    // helpers
    void computeEdgeTimes_(ChunkSize chunkSize) noexcept;
    void computeRoutes_(ChunkSize chunkSize) noexcept;  // Also builds directDeviceNeighbors_ and distanceMatrix_
    [[nodiscard]] bool canReserveRoute_(const Route& r, Time t0) const noexcept;
    void reserveRoute_(const Route& r, Time t0) noexcept;
    [[nodiscard]] bool swCapOkAt_(int sid, Time s, Time e, bool isIn) const noexcept;
    void swReserve_(int sid, Time s, Time e, bool isIn) noexcept;

    /// @brief Alpha-beta model to calculate link transfer time
    /// @param bandwidth bandwidth of the link (in GiB/sec)
    /// @param latency latency of the link (in microseconds)
    /// @param chunkSize chunk size (in bytes)
    /// @return transfer time of a chunk (in microseconds)
    [[nodiscard]] static Time alphaBetaModel_(Bandwidth bandwidth,
                                              Latency latency,
                                              ChunkSize chunkSize) noexcept;
};
}  // namespace tacos
