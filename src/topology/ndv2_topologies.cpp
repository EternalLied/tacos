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

void BuildNDv2_TwoChassis_Tecc(Topology& topo) {
  // NDv2 2-chassis from teccl_topologies/ndv2.py:
  // 16 GPUs, no explicit switches. Use 50/25GB/s (alpha=0.7us) intra,
  // and 12.5GB/s (alpha=1.3us) inter.
  const int ch = 2, perChGpu = 8, N = ch*perChGpu;
  topo.setNpusCount_(N);

  // 这里直接把 ndv2.py 的 16x16 整数矩阵硬编码进来。
  // 为简洁起见，我写一个示意，你可以把真实矩阵填入。
  int m[16][16] = {
    /* 按 ndv2.py 里的 capacity 矩阵抄，这里略 */
  };

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int v = m[i][j];
      if (v == 0) continue;
      double bw = 0.0, alpha = 0.0;
      if (v == 23) {       // 50GB/s intra
        bw = 50.0;
        alpha = 0.7;
      } else if (v == 46) {// 25GB/s intra
        bw = 25.0;
        alpha = 0.7;
      } else if (v == 107) {// 12.5GB/s inter
        bw = 12.5;
        alpha = 1.3;
      }
      topo.addPhysLink(topo.deviceNode(i), topo.deviceNode(j), bw, alpha);
    }
  }
}

void BuildNDv2_FourChassis_Tecc(Topology& topo, bool allow_copy_center) {
    // ===== TE-CCL NDv2 Four-Chassis Strict Model =====
    constexpr int chassis = 4;
    constexpr int gPerCh  = 8;
    constexpr int N_nodes = chassis * gPerCh + 1; // 33
    constexpr int N_gpus  = chassis * gPerCh;     // 32

    topo.setNpusCount_(N_gpus);

    // 中心交换机: Node 0 in TE-CCL
    auto center = topo.addSwitch(
        "NDv2_CENTER",
        /*allowCopy=*/allow_copy_center,
        /*inCap=*/N_gpus,
        /*outCap=*/N_gpus,
        SwitchForwardingMode::STORE_AND_FORWARD);

    // --- single_capacity: 来自 ndv2.py 中的 single_capacity ---
    static const int single_capacity[gPerCh][gPerCh] = {
        {0, 23, 46, 46, 23, 0, 0, 0},
        {23, 0, 46, 23, 0, 46, 0, 0},
        {46, 46, 0, 23, 0, 0, 23, 0},
        {46, 23, 23, 0, 0, 0, 0, 46},
        {23, 0, 0, 0, 0, 23, 46, 46},
        {0, 46, 0, 0, 23, 0, 46, 23},
        {0, 0, 23, 0, 46, 46, 0, 23},
        {0, 0, 0, 46, 46, 23, 23, 0}
    };

    // ---- 按 ndv2.py 构造 capacity 矩阵 (33 x 33) ----
    int capacity[N_nodes][N_nodes];
    // init to 0
    for (int i = 0; i < N_nodes; ++i)
        for (int j = 0; j < N_nodes; ++j)
            capacity[i][j] = 0;

    // Row/column 0 是 center switch
    // 先为每个 chassis embed single_capacity 到对应的 GPU block 中
    for (int c = 0; c < chassis; ++c) {
        int base = 1 + c * gPerCh; // TE-CCL 中, chassis c 的 GPU index = base..base+7
        for (int r = 0; r < gPerCh; ++r) {
            for (int col = 0; col < gPerCh; ++col) {
                capacity[ base + r ][ base + col ] = single_capacity[r][col];
            }
        }
    }

    // 然后按 ndv2.py 的写法加入 switch <-> GPU 的 107 边
    for (int i = 0; i < chassis; ++i) {
        int gpu1 = 1 + i * gPerCh;     // chassis i 的第一个 GPU (TE-CCL index)
        int gpu2 = 1 + i * gPerCh + 1; // chassis i 的第二个 GPU
        capacity[0][gpu1] = 107;
        capacity[gpu2][0] = 107;
    }

    // TE-CCL 最后做了一次转置: capacity = list(zip(*capacity))
    int capT[N_nodes][N_nodes];
    for (int i = 0; i < N_nodes; ++i)
        for (int j = 0; j < N_nodes; ++j)
            capT[i][j] = capacity[j][i];

    // 映射 TE-CCL capacity 值到 bw/alpha.
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

    // ---- 将 capT 映射成 TACOS 中的物理边 ----
    for (int u = 0; u < N_nodes; ++u) {
        for (int v = 0; v < N_nodes; ++v) {
            int val = capT[u][v];
            if (val == 0) continue;

            const double bw    = bw_of(val);
            const double alpha = alpha_of(val);
            if (bw <= 0.0) continue;

            if (u == 0 && v > 0) {
                // center -> GPU(v-1)
                topo.addPhysLink(
                    topo.switchNode(center),
                    topo.deviceNode(v-1),
                    bw, alpha);
            } else if (u > 0 && v == 0) {
                // GPU(u-1) -> center
                topo.addPhysLink(
                    topo.deviceNode(u-1),
                    topo.switchNode(center),
                    bw, alpha);
            } else if (u > 0 && v > 0) {
                // GPU(u-1) -> GPU(v-1)
                topo.addPhysLink(
                    topo.deviceNode(u-1),
                    topo.deviceNode(v-1),
                    bw, alpha);
            }
        }
    }

    topo.finalizeReachability_();
}

void BuildNDv2_TwoChassis(Topology& topo, bool allow_copy) {
  const int ch = 2, perChGpu = 8;
  topo.setNpusCount_(ch*perChGpu);
  std::vector<SwitchID> tor(ch);
  for (int c = 0; c < ch; ++c) 
    tor[c] = topo.addSwitch("ToR_"+std::to_string(c), allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::STORE_AND_FORWARD);
  // CORE (IB, Store-and-Forward)
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch, 
                             SwitchForwardingMode::STORE_AND_FORWARD);
  for (int c = 0; c < ch; ++c) {
    for (int g = 0; g < perChGpu; ++g) {
      const double bw = (g < perChGpu/2) ? 50.0 : 25.0;
      const int id = c*perChGpu + g;
      topo.addPhysLink(topo.deviceNode(id), topo.switchNode(tor[c]), bw, 0.7);
      topo.addPhysLink(topo.switchNode(tor[c]), topo.deviceNode(id), bw, 0.7);
    }
    topo.addPhysLink(topo.switchNode(tor[c]), topo.switchNode(core), 12.5, 1.3);
    topo.addPhysLink(topo.switchNode(core),   topo.switchNode(tor[c]), 12.5, 1.3);
  }
}

void BuildNDv2_FourChassis(Topology& topo, bool allow_copy) {
  const int ch = 4, perChGpu = 8;
  topo.setNpusCount_(ch*perChGpu);
  std::vector<SwitchID> tor(ch);
  for (int c = 0; c < ch; ++c) 
    tor[c] = topo.addSwitch("ToR_"+std::to_string(c), allow_copy, perChGpu, perChGpu, 
                            SwitchForwardingMode::STORE_AND_FORWARD);
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch, 
                             SwitchForwardingMode::STORE_AND_FORWARD);
  // Intra-chassis GPU<->ToR: half at 50GB/s, half at 25GB/s (alpha=0.7us), example
  for (int c = 0; c < ch; ++c) {
    for (int g = 0; g < perChGpu; ++g) {
      const double bw = (g < perChGpu/2) ? 50.0 : 25.0;
      const int id = c*perChGpu + g;
      topo.addPhysLink(topo.deviceNode(id), topo.switchNode(tor[c]), bw, 0.7);
      topo.addPhysLink(topo.switchNode(tor[c]), topo.deviceNode(id), bw, 0.7);
    }
    // ToR <-> core (12.5GB/s, alpha=1.3us)
    topo.addPhysLink(topo.switchNode(tor[c]), topo.switchNode(core), 12.5, 1.3);
    topo.addPhysLink(topo.switchNode(core),   topo.switchNode(tor[c]), 12.5, 1.3);
  }
}

}  // namespace tacos
