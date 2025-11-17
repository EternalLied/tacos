/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/topology/topology.h>

namespace tacos {

/// @brief A simple N-GPU single-switch topology using hyper-edges via the switch.
///        All GPUs connect to one logical switch with uniform per-port bandwidth/latency.
class SwitchClique final : public Topology {
 public:
  /// @param npusCount       number of GPUs
  /// @param uplinkBandwidth per-port GPU<->Switch bandwidth (GiB/sec)
  /// @param gpu2swLatency   GPU->Switch latency (us)
  /// @param sw2gpuLatency   Switch->GPU latency (us)
  /// @param maxParallel     optional cap on #simultaneous hyper-edges through the switch
  SwitchClique(int npusCount,
               Bandwidth uplinkBandwidth,
               Latency gpu2swLatency,
               Latency sw2gpuLatency,
               int maxParallel = -1) noexcept {
    setNpusCount_(npusCount);
    std::vector<NpuID> ports(npusCount);
    for (int i = 0; i < npusCount; ++i) ports[i] = i;
    addSwitchUniform(ports, uplinkBandwidth, gpu2swLatency, sw2gpuLatency, maxParallel);
  }
};

} // namespace tacos
