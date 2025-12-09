/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <tacos/topology/amd_topologies.h>
#include <vector>
#include <string>
#include <map>

namespace tacos {

void BuildAMD_MI250_2Chassis(Topology& topo, bool allow_copy) {
  // TE-CCL AMD MI250 2-chassis topology
  // Each chassis: 16 GPUs + 8 small switches (每2个GPU共享1个小交换机)
  // Small switches连接到1个中心交换机
  
  const int ch = 2, perChGpu = 16;
  const int switchPerCh = 8;  // 每个chassis 8个小交换机
  topo.setNpusCount_(ch * perChGpu);
  
  // adjacency_list定义GPU间的直连 (来自Teccl)
  std::map<int, std::map<int, int>> adjacency_list;
  adjacency_list[0] = {{1, 4}, {4, 2}, {8, 1}};
  adjacency_list[1] = {{0, 4}, {5, 1}, {9, 1}, {10, 1}};
  adjacency_list[2] = {{3, 4}, {6, 1}, {10, 1}, {9, 1}};
  adjacency_list[3] = {{2, 4}, {7, 2}, {11, 1}};
  adjacency_list[4] = {{6, 1}, {5, 4}, {0, 2}};
  adjacency_list[5] = {{1, 1}, {4, 4}, {7, 1}, {6, 1}};
  adjacency_list[6] = {{4, 1}, {7, 4}, {2, 1}, {5, 1}};
  adjacency_list[7] = {{5, 1}, {6, 4}, {3, 2}};
  adjacency_list[8] = {{0, 1}, {9, 4}, {12, 2}};
  adjacency_list[9] = {{1, 1}, {2, 1}, {8, 4}, {13, 1}};
  adjacency_list[10] = {{1, 1}, {2, 1}, {11, 4}, {14, 1}};
  adjacency_list[11] = {{3, 1}, {10, 4}, {15, 2}};
  adjacency_list[12] = {{8, 2}, {13, 4}, {14, 1}};
  adjacency_list[13] = {{9, 1}, {12, 4}, {15, 1}, {14, 1}};
  adjacency_list[14] = {{10, 1}, {12, 1}, {13, 1}, {15, 4}};
  adjacency_list[15] = {{11, 2}, {13, 1}, {14, 4}};
  
  // 根据Teccl代码: gpu_link = 50, switch_link_capacity = 25.0
  // 图片显示Gbps,但Teccl代码中直接用GB/s: 50→200(×4), 100(×2), 50(×1)
  const double gpu_link = 50.0;          // GPU间链路基础: 50 GB/s
  const double switch_link = 25.0;       // 交换机链路: 25 GB/s
  const double alpha_gpu = 0.6;          // GPU链路延迟: 0.6μs (图片标注)
  const double alpha_switch = 0.75;      // 交换机链路延迟: 0.75μs (图片标注)
  
  // 为每个chassis创建8个小交换机
  std::vector<std::vector<SwitchID>> smallSwitches(ch);
  for (int c = 0; c < ch; ++c) {
    for (int s = 0; s < switchPerCh; ++s) {
      smallSwitches[c].push_back(
        topo.addSwitch("SmallSW_C" + std::to_string(c) + "_" + std::to_string(s),
                      allow_copy, 2, 2,
                      SwitchForwardingMode::CUT_THROUGH));
    }
  }
  
  // 创建中心交换机
  auto centerSwitch = topo.addSwitch("CENTER", false, ch * switchPerCh, ch * switchPerCh,
                                    SwitchForwardingMode::CUT_THROUGH);
  
  // 为每个chassis构建拓扑
  for (int c = 0; c < ch; ++c) {
    int gpuBase = c * perChGpu;
    
    // 1. GPU间直连 (基于adjacency_list)
    for (int i = 0; i < perChGpu; ++i) {
      int gpuI = gpuBase + i;
      if (adjacency_list.count(i)) {
        for (const auto& [j, weight] : adjacency_list[i]) {
          if (j < perChGpu) {
            int gpuJ = gpuBase + j;
            double bw = gpu_link * weight;
            topo.addPhysLink(topo.deviceNode(gpuI), topo.deviceNode(gpuJ), bw, alpha_gpu);
          }
        }
      }
    }
    
    // 2. GPU连接到小交换机 (每2个GPU共享1个小交换机)
    for (int s = 0; s < switchPerCh; ++s) {
      int gpu0 = gpuBase + 2 * s;
      int gpu1 = gpuBase + 2 * s + 1;
      
      // GPU <-> SmallSwitch (双向, 25 GB/s)
      topo.addPhysLink(topo.deviceNode(gpu0), topo.switchNode(smallSwitches[c][s]), 
                      switch_link, alpha_switch);
      topo.addPhysLink(topo.switchNode(smallSwitches[c][s]), topo.deviceNode(gpu0), 
                      switch_link, alpha_switch);
      
      topo.addPhysLink(topo.deviceNode(gpu1), topo.switchNode(smallSwitches[c][s]), 
                      switch_link, alpha_switch);
      topo.addPhysLink(topo.switchNode(smallSwitches[c][s]), topo.deviceNode(gpu1), 
                      switch_link, alpha_switch);
    }

    // 3. 小交换机连接到中心交换机
    for (int s = 0; s < switchPerCh; ++s) {
      // SmallSwitch -> CenterSwitch
      topo.addPhysLink(topo.switchNode(smallSwitches[c][s]), topo.switchNode(centerSwitch),
                      switch_link, alpha_switch);
      
      // CenterSwitch -> SmallSwitch
      topo.addPhysLink(topo.switchNode(centerSwitch), topo.switchNode(smallSwitches[c][s]),
                      switch_link, alpha_switch);
    }
    
    // // 3. 小交换机连接到中心交换机 (每个方向2条并行链路，实现1:1的收敛比)
    // for (int s = 0; s < switchPerCh; ++s) {
    //   // SmallSwitch -> CenterSwitch (2条并行链路)
    //   topo.addPhysLink(topo.switchNode(smallSwitches[c][s]), topo.switchNode(centerSwitch),
    //                   switch_link, alpha_switch);
    //   topo.addPhysLink(topo.switchNode(smallSwitches[c][s]), topo.switchNode(centerSwitch),
    //                   switch_link, alpha_switch);
      
    //   // CenterSwitch -> SmallSwitch (2条并行链路)
    //   topo.addPhysLink(topo.switchNode(centerSwitch), topo.switchNode(smallSwitches[c][s]),
    //                   switch_link, alpha_switch);
    //   topo.addPhysLink(topo.switchNode(centerSwitch), topo.switchNode(smallSwitches[c][s]),
    //                   switch_link, alpha_switch);
    // }
  }
}

}  // namespace tacos
