/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <tacos/topology/amd_topologies.h>
#include <vector>
#include <string>

namespace tacos {

void BuildAMD_MI250_2Chassis(Topology& topo, bool allow_copy) {
  const int ch = 2, perChGpu = 16;
  topo.setNpusCount_(ch*perChGpu);
  // Leaf switch + aggregation switch per chassis (IB, Store-and-Forward)
  std::vector<SwitchID> leaf(ch), agg(ch);
  for (int c = 0; c < ch; ++c) {
    leaf[c] = topo.addSwitch("Leaf_"+std::to_string(c), allow_copy, perChGpu, perChGpu, 
                             SwitchForwardingMode::STORE_AND_FORWARD);
    agg[c]  = topo.addSwitch("Agg_"+std::to_string(c),  allow_copy, perChGpu, perChGpu, 
                             SwitchForwardingMode::STORE_AND_FORWARD);
    // GPU<->Leaf: 200/100/50 GB/s, alpha=0.6us (three categories)
    for (int g = 0; g < perChGpu; ++g) {
      double bw = (g<4)?200.0 : (g<8)?100.0 : 50.0;
      const int id = c*perChGpu + g;
      topo.addPhysLink(topo.deviceNode(id), topo.switchNode(leaf[c]), bw, 0.6);
      topo.addPhysLink(topo.switchNode(leaf[c]), topo.deviceNode(id), bw, 0.6);
    }
    // Leaf<->Agg (25GB/s, alpha=0.75)
    topo.addPhysLink(topo.switchNode(leaf[c]), topo.switchNode(agg[c]), 25.0, 0.75);
    topo.addPhysLink(topo.switchNode(agg[c]),  topo.switchNode(leaf[c]), 25.0, 0.75);
  }
  // Inter-chassis Agg<->Agg (25GB/s, alpha=0.75)
  topo.addPhysLink(topo.switchNode(agg[0]), topo.switchNode(agg[1]), 25.0, 0.75);
  topo.addPhysLink(topo.switchNode(agg[1]), topo.switchNode(agg[0]), 25.0, 0.75);
}

void BuildAMD_MI250_4Chassis(Topology& topo, bool allow_copy) {
  // Extendable from 2ch: Agg[i] <-> Core 25GB/s, alpha=0.75, then Core interconnect
  // AMD MI250 uses multi-chassis IB (Store-and-Forward)
  const int ch = 4, perChGpu = 16;
  topo.setNpusCount_(ch*perChGpu);
  std::vector<SwitchID> leaf(ch), agg(ch);
  for (int c = 0; c < ch; ++c) {
    leaf[c] = topo.addSwitch("Leaf_"+std::to_string(c), allow_copy, perChGpu, perChGpu, 
                             SwitchForwardingMode::STORE_AND_FORWARD);
    agg[c]  = topo.addSwitch("Agg_"+std::to_string(c),  allow_copy, perChGpu, perChGpu, 
                             SwitchForwardingMode::STORE_AND_FORWARD);
    for (int g = 0; g < perChGpu; ++g) {
      double bw = (g<4)?200.0 : (g<8)?100.0 : 50.0;
      const int id = c*perChGpu + g;
      topo.addPhysLink(topo.deviceNode(id), topo.switchNode(leaf[c]), bw, 0.6);
      topo.addPhysLink(topo.switchNode(leaf[c]), topo.deviceNode(id), bw, 0.6);
    }
    topo.addPhysLink(topo.switchNode(leaf[c]), topo.switchNode(agg[c]), 25.0, 0.75);
    topo.addPhysLink(topo.switchNode(agg[c]),  topo.switchNode(leaf[c]), 25.0, 0.75);
  }
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch, 
                             SwitchForwardingMode::STORE_AND_FORWARD);
  for (int c = 0; c < ch; ++c) {
    topo.addPhysLink(topo.switchNode(agg[c]), topo.switchNode(core), 25.0, 0.75);
    topo.addPhysLink(topo.switchNode(core),   topo.switchNode(agg[c]), 25.0, 0.75);
  }
}

}  // namespace tacos
