/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <tacos/topology/dgx_topologies.h>
#include <string>
#include <vector>
#include <utility>

namespace tacos {

void BuildDGX1_Tecc(Topology& topo, double alpha_us) {
  // TE-CCL DGX1: 8 GPUs, direct GPU-GPU graph, no explicit switch.
  // Each "1" or "2" in the adjacency matrix corresponds to 25 or 50 GB/s.
  const int N = 8;
  topo.setNpusCount_(N);

  auto add = [&](int i, int j, int k) {
    if (k == 0) return;
    const double bw = 25.0 * k; // 25GB/s per NVLink
    topo.addPhysLink(topo.deviceNode(i), topo.deviceNode(j), bw, alpha_us);
    topo.addPhysLink(topo.deviceNode(j), topo.deviceNode(i), bw, alpha_us);
  };

  // adjacency from teccl/dgx1.py (symmetric):
  // Row 0: to 1(2),2(1),3(1),4(2)
  add(0,1,2); add(0,2,1); add(0,3,1); add(0,4,2);
  // Row 1: to 2(1),3(2),5(1)
  add(1,2,1); add(1,3,2); add(1,5,1);
  // Row 2: to 3(2),6(2)
  add(2,3,2); add(2,6,2);
  // Row 3: to 7(1)
  add(3,7,1);
  // Row 4: to 5(2),6(1),7(1)
  add(4,5,2); add(4,6,1); add(4,7,1);
  // Row 5: to 6(1),7(2)
  add(5,6,1); add(5,7,2);
  // Row 6: to 7(2)
  add(6,7,2);
}

void BuildDGX1_SingleChassis(Topology& topo,
                             double bw_gbps,
                             double alpha_us,
                             bool allow_copy) {
  const int N = 8;
  topo.setNpusCount_(N);
  auto sw = topo.addSwitch("NVSW", allow_copy, /*inCap*/N, /*outCap*/N, 
                           SwitchForwardingMode::CUT_THROUGH);
  for (int g = 0; g < N; ++g) {
    topo.addPhysLink(topo.deviceNode(g), topo.switchNode(sw), bw_gbps, alpha_us);
    topo.addPhysLink(topo.switchNode(sw), topo.deviceNode(g), bw_gbps, alpha_us);
  }
}

void BuildDGX2_TwoChassis_Tecc(Topology& topo, bool allow_copy) {
  // TE-CCL DGX2 two-chassis (Fig.13): 2*16 GPUs, each chassis has one logical NVSW.
  // Intra-chassis: GPU<->NVSW 125GB/s, alpha=0.35us.
  // Inter-chassis: selected GPU<->GPU edges 12.5GB/s, alpha=2.6us.
  const int perChGpu = 16;
  const int numCh = 2;
  topo.setNpusCount_(numCh * perChGpu);

  // one NVSwitch per chassis (we keep it mainly for on-node modeling)
  SwitchID sw[2];
  for (int c = 0; c < numCh; ++c) {
    sw[c] = topo.addSwitch("NVSW_"+std::to_string(c),
                           allow_copy,
                           perChGpu, perChGpu,
                           SwitchForwardingMode::CUT_THROUGH);
  }

  // Intra-chassis: star GPU<->NVSW with 125GB/s & 0.35us
  for (int c = 0; c < numCh; ++c) {
    for (int g = 0; g < perChGpu; ++g) {
      int gpu = c*perChGpu + g;
      topo.addPhysLink(topo.deviceNode(gpu), topo.switchNode(sw[c]),
                       125.0, 0.35);
      topo.addPhysLink(topo.switchNode(sw[c]), topo.deviceNode(gpu),
                       125.0, 0.35);
    }
  }

  // Inter-chassis: GPU->GPU direct links, 12.5GB/s, alpha=2.6us
  // Mapping from teccl_topologies/dgx2.py (taccl map: {"1":[0], "3":[2], ... "15":[14]})
  std::vector<std::pair<int,int>> pairs = {
    {1,0}, {3,2}, {5,4}, {7,6}, {9,8}, {11,10}, {13,12}, {15,14}
  };

  auto connect_gpu_pair = [&](int gA, int gB) {
    topo.addPhysLink(topo.deviceNode(gA), topo.deviceNode(gB),
                     12.5, 2.6);
    topo.addPhysLink(topo.deviceNode(gB), topo.deviceNode(gA),
                     12.5, 2.6);
  };

  // chassis 0 GPUs [0..15], chassis 1 GPUs [16..31]
  for (auto [sLocal, rLocal] : pairs) {
    int s0 = sLocal;               // chassis0 sender
    int r1 = perChGpu + rLocal;    // chassis1 receiver
    int s1 = perChGpu + sLocal;    // chassis1 sender
    int r0 = rLocal;               // chassis0 receiver
    connect_gpu_pair(s0, r1);
    connect_gpu_pair(s1, r0);
  }
}

void BuildDGX2_TwoChassis(Topology& topo, bool allow_copy) {
  const int perChGpu = 16;
  topo.setNpusCount_(2*perChGpu);
  // two chassis NVSwitch blocks (Cut-Through)
  auto swA = topo.addSwitch("NVSW_A", allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::CUT_THROUGH);
  auto swB = topo.addSwitch("NVSW_B", allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::CUT_THROUGH);
  // core/aggregation switch (IB, Store-and-Forward)
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/8, /*out*/8, 
                             SwitchForwardingMode::STORE_AND_FORWARD);

  // Intra-chassis: GPU<->NVSW 125 GB/s α=0.35us  (TE-CCL Fig.13)
  for (int g = 0; g < perChGpu; ++g) {
    topo.addPhysLink(topo.deviceNode(g),               topo.switchNode(swA), 125.0, 0.35);
    topo.addPhysLink(topo.switchNode(swA), topo.deviceNode(g),               125.0, 0.35);
  }
  for (int g = 0; g < perChGpu; ++g) {
    const int id = perChGpu + g;
    topo.addPhysLink(topo.deviceNode(id),              topo.switchNode(swB), 125.0, 0.35);
    topo.addPhysLink(topo.switchNode(swB), topo.deviceNode(id),              125.0, 0.35);
  }
  // Inter-chassis via core: 100 GB/s α=2.6us (both directions)
  topo.addPhysLink(topo.switchNode(swA), topo.switchNode(core), 100, 2.60);
  topo.addPhysLink(topo.switchNode(core),topo.switchNode(swA),  100, 2.60);
  topo.addPhysLink(topo.switchNode(swB), topo.switchNode(core), 100, 2.60);
  topo.addPhysLink(topo.switchNode(core),topo.switchNode(swB),  100, 2.60);
}

}  // namespace tacos
