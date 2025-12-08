/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <tacos/topology/ndv2_topologies.h>
#include <vector>
#include <string>

namespace tacos {

void BuildNDv2_TwoChassis(Topology& topo, bool allow_copy) {
  const int ch = 2, perChGpu = 8;
  topo.setNpusCount_(ch*perChGpu);
  
  // single_capacity 矩阵: 定义每个chassis内GPU间的直连带宽
  static const int single_capacity[perChGpu][perChGpu] = {
      {0, 23, 46, 46, 23, 0, 0, 0},
      {23, 0, 46, 23, 0, 46, 0, 0},
      {46, 46, 0, 23, 0, 0, 23, 0},
      {46, 23, 23, 0, 0, 0, 0, 46},
      {23, 0, 0, 0, 0, 23, 46, 46},
      {0, 46, 0, 0, 23, 0, 46, 23},
      {0, 0, 23, 0, 46, 46, 0, 23},
      {0, 0, 0, 46, 46, 23, 23, 0}
  };
  
  auto bw_of = [](int v)->double {
      if (v == 23)  return 50.0;  // 50 GB/s
      if (v == 46)  return 25.0;  // 25 GB/s
      if (v == 107) return 12.5;  // 12.5 GB/s
      return 0.0;
  };
  auto alpha_of = [](int v)->double {
      if (v == 23 || v == 46) return 0.7;
      if (v == 107)           return 1.3;
      return 0.0;
  };
  
  // 为每个chassis添加intra-chassis GPU直连
  for (int c = 0; c < ch; ++c) {
    int base = c * perChGpu;
    for (int r = 0; r < perChGpu; ++r) {
      for (int col = 0; col < perChGpu; ++col) {
        int val = single_capacity[r][col];
        if (val == 0) continue;
        
        double bw = bw_of(val);
        double alpha = alpha_of(val);
        if (bw <= 0.0) continue;
        
        topo.addPhysLink(
          topo.deviceNode(base + r),
          topo.deviceNode(base + col),
          bw, alpha);
      }
    }
  }
  
  // 中心交换机 (CUT_THROUGH, 连接chassis间)
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch, 
                             SwitchForwardingMode::CUT_THROUGH);
  
  // 每个chassis的GPU 0和GPU 1连接到中心交换机 (单向连接,对齐Teccl)
  // Teccl: capacity[0][i*8+1]=107 (switch→GPU0), capacity[i*8+2][0]=107 (GPU1→switch)
  for (int c = 0; c < ch; ++c) {
    int gpu0 = c * perChGpu;      // chassis c 的 GPU 0
    int gpu1 = c * perChGpu + 1;  // chassis c 的 GPU 1
    
    // Switch → GPU 0 (单向: 用于接收跨chassis数据)
    topo.addPhysLink(topo.switchNode(core), topo.deviceNode(gpu0), 12.5, 1.3);
    
    // GPU 1 → Switch (单向: 用于发送跨chassis数据)
    topo.addPhysLink(topo.deviceNode(gpu1), topo.switchNode(core), 12.5, 1.3);
  }
}

void BuildNDv2_FourChassis(Topology& topo, bool allow_copy) {
  const int ch = 4, perChGpu = 8;
  topo.setNpusCount_(ch*perChGpu);
  
  // single_capacity 矩阵: 定义每个chassis内GPU间的直连带宽
  static const int single_capacity[perChGpu][perChGpu] = {
      {0, 23, 46, 46, 23, 0, 0, 0},
      {23, 0, 46, 23, 0, 46, 0, 0},
      {46, 46, 0, 23, 0, 0, 23, 0},
      {46, 23, 23, 0, 0, 0, 0, 46},
      {23, 0, 0, 0, 0, 23, 46, 46},
      {0, 46, 0, 0, 23, 0, 46, 23},
      {0, 0, 23, 0, 46, 46, 0, 23},
      {0, 0, 0, 46, 46, 23, 23, 0}
  };
  
  auto bw_of = [](int v)->double {
      if (v == 23)  return 50.0;  // 50 GB/s
      if (v == 46)  return 25.0;  // 25 GB/s
      if (v == 107) return 12.5;  // 12.5 GB/s
      return 0.0;
  };
  auto alpha_of = [](int v)->double {
      if (v == 23 || v == 46) return 0.7;
      if (v == 107)           return 1.3;
      return 0.0;
  };
  
  // 为每个chassis添加intra-chassis GPU直连
  for (int c = 0; c < ch; ++c) {
    int base = c * perChGpu;
    for (int r = 0; r < perChGpu; ++r) {
      for (int col = 0; col < perChGpu; ++col) {
        int val = single_capacity[r][col];
        if (val == 0) continue;
        
        double bw = bw_of(val);
        double alpha = alpha_of(val);
        if (bw <= 0.0) continue;
        
        topo.addPhysLink(
          topo.deviceNode(base + r),
          topo.deviceNode(base + col),
          bw, alpha);
      }
    }
  }
  
  // 中心交换机 (CUT_THROUGH, 连接chassis间)
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch, 
                             SwitchForwardingMode::CUT_THROUGH);
  
  // 每个chassis的GPU 0和GPU 1连接到中心交换机 (单向连接,对齐Teccl)
  // Teccl: capacity[0][i*8+1]=107 (switch→GPU0), capacity[i*8+2][0]=107 (GPU1→switch)
  for (int c = 0; c < ch; ++c) {
    int gpu0 = c * perChGpu;      // chassis c 的 GPU 0
    int gpu1 = c * perChGpu + 1;  // chassis c 的 GPU 1
    
    // Switch → GPU 0 (单向: 用于接收跨chassis数据)
    topo.addPhysLink(topo.switchNode(core), topo.deviceNode(gpu0), 12.5, 1.3);
    
    // GPU 1 → Switch (单向: 用于发送跨chassis数据)
    topo.addPhysLink(topo.deviceNode(gpu1), topo.switchNode(core), 12.5, 1.3);
  }
}

}  // namespace tacos
