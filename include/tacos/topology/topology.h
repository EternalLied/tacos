/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <cstdint>
#include <tacos/event_queue/event_queue.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>
#include <string>
#include <tuple>

namespace tacos {

  /// Node kind in the physical graph
  enum class NodeKind { Device, Switch };

  /// @brief Switch forwarding mode
  /// STORE_AND_FORWARD: IB, Ethernet RoCE, multi-chassis IB, multi-tier clos
  ///   - alpha = alpha1 + alpha2 (cumulative)
  ///   - beta = beta1 + beta2 (cumulative)
  /// CUT_THROUGH: NVSwitch (DGX-2, DGX A100, DGX H100 NVL, GH200 NVLink Switch)
  ///   - alpha = alphaNVLink (counted once)
  ///   - beta = max(beta1, beta2) (bottleneck, pipelined)
  ///   - Acts as transparent "smart wire", equivalent to 1 hop
  enum class SwitchForwardingMode { 
    STORE_AND_FORWARD,  // Default: cumulative latency and bandwidth
    CUT_THROUGH         // NVSwitch: pipelined, single-hop equivalent
  };

  /// @brief Switch identifier used by hyper-edge modeling
  using SwitchID = int;
  using NodeIndex = int;  // index into [0, totalNodes_)

  // NOTE: npusCount() counts only GPUs; switches are not counted as NPUs.

class Topology {
  public:
    // data types
    using Time = EventQueue::Time;

    /// @brief NPU ID
    using NpuID = int;

    /// @brief link bandwidth: GiB/sec
    using Bandwidth = double;

    /// @brief link latency: microseconds (us)
    using Latency = double;

    // /// @brief Time unit: microseconds (TEN uses this unit)
    // using Time = long long;    

    struct Switch {
      std::string name;
      bool  allowCopy = false;
      int   inCap = -1;   // default = degree
      int   outCap = -1;  // default = degree
      SwitchForwardingMode forwardingMode = SwitchForwardingMode::STORE_AND_FORWARD;
    };

    // ====== Multi-switch physical graph (Switch-Transit) ======
    struct PhysLink {
      NodeIndex src;   // device or switch node index in [0,totalNodes)
      NodeIndex dst;   // device or switch node index in [0,totalNodes)
      Bandwidth bw;    // GiB/s (per-direction)
      Latency   alpha; // μs per transfer (per-direction)
    };

    /// @brief Reset device count (GPUs). Switches can be added afterwards.
    void setNpusCount_(int npusCount) noexcept;

    /// @brief Add a new switch node; returns SwitchID
    SwitchID addSwitch(const std::string& name, bool allowCopy=false,
                      int inCap=-1, int outCap=-1, 
                      SwitchForwardingMode mode=SwitchForwardingMode::STORE_AND_FORWARD) noexcept;

    /// @brief Map (device id | switch id) to node index in physical graph
    [[nodiscard]] NodeIndex deviceNode(NpuID d) const noexcept { return d; }
    [[nodiscard]] NodeIndex switchNode(SwitchID s) const noexcept { return npusCount_ + s; }

    /// @brief total nodes = npus + switches
    [[nodiscard]] int totalNodes() const noexcept { return npusCount_ + switchesCount_; }
    [[nodiscard]] int switchesCount() const noexcept { return switchesCount_; }

    /// @brief add a physical directed link between any two nodes (device/switch)
    void addPhysLink(NodeIndex u, NodeIndex v, Bandwidth bw, Latency alpha) noexcept;

    /// @brief physical adjacency for routing (list of edges)
    const std::vector<PhysLink>& physLinks() const noexcept { return physLinks_; }

    /// @brief Get total number of physical links in the topology
    [[nodiscard]] int physLinksCount() const noexcept { return static_cast<int>(physLinks_.size()); }

    /// @brief Get link statistics by type
    /// @return tuple of (device-to-device, device-to-switch, switch-to-device, switch-to-switch) link counts
    [[nodiscard]] std::tuple<int, int, int, int> getLinkStatistics() const noexcept;

    /// @brief switch properties
    const Switch& switchAt(SwitchID sid) const noexcept { return switches_[sid]; }

    /// @brief per-switch caps (resolved defaults)
    int switchInCapResolved(SwitchID sid) const noexcept;
    int switchOutCapResolved(SwitchID sid) const noexcept;

    /// @brief finalize GPU→GPU reachability from physical graph (call after adding all links)
    void finalizeReachability_() noexcept;

    /// @brief helper: given node index, return switch id or -1 if not a switch
    [[nodiscard]] int switchIdFromNode(NodeIndex n) const noexcept {
      if (n < 0 || n >= totalNodes()) return -1;
      if (n >= npusCount_) return (n - npusCount_);
      return -1;
    }

    /// @brief Default constructor
    Topology() noexcept;

    /// @brief Get the bandwidth of a link
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return bandwidth of the link
    [[nodiscard]] Bandwidth bandwidth(NpuID src, NpuID dest) const noexcept;

    /// @brief Get the latency of a link
    /// @param src source NPU ID
    /// @param dest destination NPU ID
    /// @return latency of the link
    [[nodiscard]] Latency latency(NpuID src, NpuID dest) const noexcept;

    /// @brief Backtrack a destination NPU to find all NPUs that can send a chunk to it
    /// @param dest destination NPU ID
    /// @return set of source NPU IDs that can send a chunk to the destination NPU
    [[nodiscard]] std::vector<NpuID> backtrack(NpuID dest) const noexcept;

    /// @brief Get the number of NPUs in the topology
    /// @return number of NPUs
    [[nodiscard]] int npusCount() const noexcept;

    /// @brief Check if a link exists between two NPUs
    /// @param src src NPU ID
    /// @param dest dest NPU ID
    /// @return true if a link exists, false otherwise
    [[nodiscard]] bool connected(NpuID src, NpuID dest) const noexcept;

    // // ===== Switch-aware (TE-CCL style) APIs =====
    // /// @brief Register a switch with a set of GPU ports and create GPU->GPU hyper-edges via this switch.
    // ///        Each pair (u,v) in ports gets a synthetic link whose bandwidth equals uplinkBandwidth
    // ///        and latency equals (gpu2swLatency + sw2gpuLatency). Direct links (if any) are kept.
    // /// @param ports             GPUs attached to this switch (NPU IDs)
    // /// @param uplinkBandwidth   Per-port bandwidth (GiB/sec)
    // /// @param gpu2swLatency     GPU->Switch latency (us)
    // /// @param sw2gpuLatency     Switch->GPU latency (us)
    // /// @param maxParallelEdges  Optional cap on #simultaneous hyper-edges through this switch
    // /// @return SwitchID
    // SwitchID addSwitchUniform(const std::vector<NpuID>& ports,
    //                           Bandwidth uplinkBandwidth,
    //                           Latency gpu2swLatency,
    //                           Latency sw2gpuLatency,
    //                           int maxParallelEdges = -1) noexcept;

    // /// @brief Total count of switches registered in this topology
    // [[nodiscard]] int switchesCount() const noexcept { return switchesCount_; }

    // /// @brief Upper bound on simultaneous hyper-edges through a switch
    // [[nodiscard]] int switchParallelLimit(SwitchID sid) const noexcept;

    // /// @brief Whether src->dest is a synthetic hyper-edge via some switch
    // [[nodiscard]] bool isViaSwitch(NpuID src, NpuID dest) const noexcept;

    // /// @brief SwitchID for the hyper-edge (src,dst); -1 if not via switch
    // [[nodiscard]] int viaSwitchId(NpuID src, NpuID dest) const noexcept;

  protected:
    /// @brief number of NPUs in the topology
    int npusCount_ = -1;
    int switchesCount_ = 0;

    // /// @brief set the number of NPUs in the topology
    // /// @param npusCount number of NPUs
    // void setNpusCount_(int npusCount) noexcept;

    /// @brief Establish a connection (i.e., add a link) between two NPUs
    /// @param src src NPU
    /// @param dest dest NPU
    /// @param bandwidth bandwidth of the link (in GiB/sec)
    /// @param latency latency of the link (in microseconds)
    /// @param bidirectional if true, create dest -> src link as well
    void connect_(NpuID src,
                  NpuID dest,
                  Bandwidth bandwidth,
                  Latency latency,
                  bool bidirectional = false) noexcept;

  private:
    /// @brief true if a link exists between src and dest
    std::vector<std::vector<bool>> connected_ = {};

    /// @brief link bandwidth of src -> dest (in GiB/sec)
    std::vector<std::vector<Bandwidth>> bandwidths_ = {};

    /// @brief link latency of src -> dest (in microseconds)
    std::vector<std::vector<Latency>> latencies_ = {};

    /// @brief set of NPUs that can send a chunk to a given NPU
    std::unordered_map<NpuID, std::vector<NpuID>> backtrackMap_ = {};

    // ===== physical graph (multi-switch) =====
    std::vector<Switch> switches_ = {};
    std::vector<PhysLink> physLinks_ = {};

    // // ===== Switch-aware (TE-CCL style) data =====
    // /// @brief number of registered switches
    // int switchesCount_ = 0;

    // /// @brief hyper-edge switch id: -1 if not via switch
    // std::vector<std::vector<int>> viaSwitchId_ = {};

    // /// @brief per-switch concurrent hyper-edge cap
    // std::vector<int> switchMaxParallel_ = {};

};
}  // namespace tacos
