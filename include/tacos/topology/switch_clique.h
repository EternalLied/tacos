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
  /// @param allowCopy       whether switch allows packet copy
  /// @param maxParallel     optional cap on #simultaneous transfers through the switch (-1 = unlimited)
  SwitchClique(int npusCount,
               Bandwidth uplinkBandwidth,
               Latency gpu2swLatency,
               Latency sw2gpuLatency,
               bool allowCopy = false,
               int maxParallel = -1) noexcept {
    setNpusCount_(npusCount);
    
    // Add a single central switch connecting all GPUs
    auto sw = addSwitch("SW", allowCopy, 
                       maxParallel > 0 ? maxParallel : npusCount,  // inCap
                       maxParallel > 0 ? maxParallel : npusCount); // outCap
    
    // Connect each GPU to the switch bidirectionally
    for (int g = 0; g < npusCount; ++g) {
      addPhysLink(deviceNode(g), switchNode(sw), uplinkBandwidth, gpu2swLatency);
      addPhysLink(switchNode(sw), deviceNode(g), uplinkBandwidth, sw2gpuLatency);
    }
    
    // Finalize reachability to compute GPU->GPU paths
    finalizeReachability_();
  }
};

} // namespace tacos
