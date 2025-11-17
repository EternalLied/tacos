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
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return chunk transfer time in microseconds (us)
    [[nodiscard]] Time linkTransferTime(NpuID src, NpuID dest) const noexcept;

    /// @brief Backtrack the TEN and return the list of available source NPUs to dest
    /// @param dest destination NPU ID
    /// @return list of source NPUs available at current timestep
    std::unordered_set<NpuID> backtrack(NpuID dest) noexcept;

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

    /// @brief link transfer time of a chunk using alpha-beta model (in microseconds)
    std::vector<std::vector<Time>> linkTransferTimes_ = {};

    // ===== Multi-switch physical resources =====
    struct EdgeKey { int u; int v; };
    // busy-until on physical directed edges (device/switch graph)
    std::vector<std::vector<Time>> edgeBusyUntil_; // sized [totalNodes_][totalNodes_], -1 if no edge
    std::vector<std::vector<Time>> edgeDelta_;     // per-edge α+β·n (μs); -1 if no edge
    std::vector<std::vector<char>> hasEdge_;       // quick check

    // per-switch in/out concurrent usage calendars: intervals [start,end)
    struct Interval { Time s, e; };
    std::vector<std::vector<Interval>> swInUse_;   // [sid] list of intervals
    std::vector<std::vector<Interval>> swOutUse_;  // [sid] list of intervals
    std::vector<int> swInCap_;   // resolved caps
    std::vector<int> swOutCap_;
    std::vector<char> swAllowCopy_;

    // precomputed shortest route for each GPU pair at current chunkSize
    struct Route {
      std::vector<int> nodes;     // node indices u->...->v
      std::vector<Time> deltas;   // per-edge Δ
      Time total = 0;
    };
    std::vector<std::vector<Route>> routes_; // [srcGPU][dstGPU]

    // helpers
    void computeEdgeTimes_(ChunkSize chunkSize) noexcept;
    void computeRoutes_(ChunkSize chunkSize) noexcept;
    [[nodiscard]] bool canReserveRoute_(const Route& r, Time t0) const noexcept;
    void reserveRoute_(const Route& r, Time t0) noexcept;
    [[nodiscard]] bool swCapOkAt_(int sid, Time s, Time e, bool isIn) const noexcept;
    void swReserve_(int sid, Time s, Time e, bool isIn) noexcept;

    // // ===== Switch-aware (TE-CCL style) data =====
    // /// @brief hyper-edge switch id for link src->dest, -1 if not via switch
    // std::vector<std::vector<int>> viaSwitchId_ = {};

    // /// @brief per-switch active concurrent hyper-edges
    // std::vector<int> switchActive_ = {};

    // /// @brief per-switch cap on concurrent hyper-edges
    // std::vector<int> switchMaxParallel_ = {};

    // /// @brief helper: whether a link is constrained by a switch and still under cap
    // [[nodiscard]] bool switchCapOk_(NpuID src, NpuID dest) const noexcept;

    /// @brief Set the chunk size for the alpha-beta model
    /// @param chunkSize chunk size in bytes
    void computeLinkTimes_(ChunkSize chunkSize) noexcept;

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
