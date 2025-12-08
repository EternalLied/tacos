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

void BuildDGX1(Topology& topo, double alpha_us) {
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

void BuildDGX2_TwoChassis(Topology& topo, bool allow_copy) {
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
  // 
  // EXPERIMENT: Test unidirectional links like Teccl to understand scheduling behavior
  // In Teccl, the matrix is unidirectional but achieves bidirectional connectivity through
  // separate iterations. Here we test if TACOS can handle multi-hop routing with unidirectional links.
  std::vector<std::pair<int,int>> pairs = {
    {1,0}, {3,2}, {5,4}, {7,6}, {9,8}, {11,10}, {13,12}, {15,14}
  };

  // Test 1: Bidirectional (original TACOS approach) - UNCOMMENT to enable
  auto connect_gpu_pair_bidirectional = [&](int gA, int gB) {
    topo.addPhysLink(topo.deviceNode(gA), topo.deviceNode(gB), 12.5, 2.6);
    topo.addPhysLink(topo.deviceNode(gB), topo.deviceNode(gA), 12.5, 2.6);
  };

  // Test 2: Unidirectional (Teccl-style) - UNCOMMENT to enable
  auto connect_gpu_pair_unidirectional = [&](int gA, int gB) {
    // Only create gA -> gB direction
    topo.addPhysLink(topo.deviceNode(gA), topo.deviceNode(gB), 12.5, 2.6);
  };

  // chassis 0 GPUs [0..15], chassis 1 GPUs [16..31]
  for (auto [sLocal, rLocal] : pairs) {
    int s0 = sLocal;               // chassis0 sender
    int r1 = perChGpu + rLocal;    // chassis1 receiver
    int s1 = perChGpu + sLocal;    // chassis1 sender
    int r0 = rLocal;               // chassis0 receiver
    
    // Test unidirectional links with multi-hop AllGather support
    connect_gpu_pair_unidirectional(s0, r1);
    connect_gpu_pair_unidirectional(s1, r0);
  }
}

void BuildDGX2_TwoChassis_typeA(Topology& topo, bool allow_copy) {
  const int perChGpu = 16;
  const int totalGpu = 2 * perChGpu;
  topo.setNpusCount_(totalGpu);
  
  // Two chassis NVSwitch blocks (Cut-Through, for intra-chassis)
  auto swA = topo.addSwitch("NVSW_A", allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::CUT_THROUGH);
  auto swB = topo.addSwitch("NVSW_B", allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::CUT_THROUGH);
  
  // Single shared IB switch connecting all 32 GPUs (16 ports, CUT_THROUGH)
  // This models a centralized inter-chassis switch with 16 concurrent connections
  auto ibShared = topo.addSwitch("IB_Shared", /*allow_copy*/false, /*in*/16, /*out*/16, 
                                 SwitchForwardingMode::CUT_THROUGH);

  // Intra-chassis: GPU<->NVSW 125 GB/s α=0.35us
  for (int g = 0; g < perChGpu; ++g) {
    topo.addPhysLink(topo.deviceNode(g), topo.switchNode(swA), 125.0, 0.35);
    topo.addPhysLink(topo.switchNode(swA), topo.deviceNode(g), 125.0, 0.35);
  }
  for (int g = 0; g < perChGpu; ++g) {
    const int id = perChGpu + g;
    topo.addPhysLink(topo.deviceNode(id), topo.switchNode(swB), 125.0, 0.35);
    topo.addPhysLink(topo.switchNode(swB), topo.deviceNode(id), 125.0, 0.35);
  }
  
  // Inter-chassis: All GPUs connect to shared IB switch, 12.5 GB/s α=2.6us
  for (int g = 0; g < totalGpu; ++g) {
    topo.addPhysLink(topo.deviceNode(g), topo.switchNode(ibShared), 12.5, 2.60);
    topo.addPhysLink(topo.switchNode(ibShared), topo.deviceNode(g), 12.5, 2.60);
  }
}

void BuildDGX2_TwoChassis_typeB(Topology& topo, bool allow_copy) {
  const int perChGpu = 16;
  topo.setNpusCount_(2*perChGpu);
  // two chassis NVSwitch blocks (Cut-Through)
  auto swA = topo.addSwitch("NVSW_A", allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::CUT_THROUGH);
  auto swB = topo.addSwitch("NVSW_B", allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::CUT_THROUGH);
  
  // Two IB switches for inter-chassis (one per chassis, 8 ports each)
  // Using CUT_THROUGH mode for IB switches as requested
  auto ibA = topo.addSwitch("IB_A", /*allow_copy*/false, /*in*/8, /*out*/8, 
                            SwitchForwardingMode::CUT_THROUGH);
  auto ibB = topo.addSwitch("IB_B", /*allow_copy*/false, /*in*/8, /*out*/8, 
                            SwitchForwardingMode::CUT_THROUGH);

  // Intra-chassis: GPU<->NVSW 125 GB/s α=0.35us
  for (int g = 0; g < perChGpu; ++g) {
    topo.addPhysLink(topo.deviceNode(g), topo.switchNode(swA), 125.0, 0.35);
    topo.addPhysLink(topo.switchNode(swA), topo.deviceNode(g), 125.0, 0.35);
  }
  for (int g = 0; g < perChGpu; ++g) {
    const int id = perChGpu + g;
    topo.addPhysLink(topo.deviceNode(id), topo.switchNode(swB), 125.0, 0.35);
    topo.addPhysLink(topo.switchNode(swB), topo.deviceNode(id), 125.0, 0.35);
  }
  
  // Inter-chassis: GPU<->IB connections, 12.5 GB/s α=2.6us
  // Chassis A GPUs connect to IB_A
  for (int g = 0; g < perChGpu; ++g) {
    topo.addPhysLink(topo.deviceNode(g), topo.switchNode(ibA), 12.5, 2.60);
    topo.addPhysLink(topo.switchNode(ibA), topo.deviceNode(g), 12.5, 2.60);
  }
  // Chassis B GPUs connect to IB_B
  for (int g = 0; g < perChGpu; ++g) {
    const int id = perChGpu + g;
    topo.addPhysLink(topo.deviceNode(id), topo.switchNode(ibB), 12.5, 2.60);
    topo.addPhysLink(topo.switchNode(ibB), topo.deviceNode(id), 12.5, 2.60);
  }
  
  // IB switches interconnection: 8 parallel bidirectional links (12.5 GB/s each)
  // This allows true concurrent transmission across multiple physical links
  // Total aggregate bandwidth: 8 × 12.5 = 100 GB/s per direction
  for (int i = 0; i < 8; ++i) {
    topo.addPhysLink(topo.switchNode(ibA), topo.switchNode(ibB), 12.5, 2.60);
    topo.addPhysLink(topo.switchNode(ibB), topo.switchNode(ibA), 12.5, 2.60);
  }
}

}  // namespace tacos
