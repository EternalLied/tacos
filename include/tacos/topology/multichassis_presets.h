#pragma once
#include <tacos/topology/topology.h>
#include <vector>
#include <string>

namespace tacos {

// ---- DGX-1: 8 GPUs, single-chassis, NVSwitch approximated as 1-level switch (example approximation) ----
// Internal link example: 125 GB/s, alpha=0.35us (adjust based on actual hardware)
inline void BuildDGX1_SingleChassis(Topology& topo,
                                    double bw_gbps = 125.0,
                                    double alpha_us = 0.35,
                                    bool allow_copy=false) {
  const int N = 8;
  topo.setNpusCount_(N);
  auto sw = topo.addSwitch("NVSW", allow_copy, /*inCap*/N, /*outCap*/N);
  for (int g = 0; g < N; ++g) {
    topo.addPhysLink(topo.deviceNode(g), topo.switchNode(sw), bw_gbps, alpha_us);
    topo.addPhysLink(topo.switchNode(sw), topo.deviceNode(g), bw_gbps, alpha_us);
  }
  topo.finalizeReachability_();
}

// ---- DGX-2: Two-chassis example (ref. TE-CCL Fig.13: GPU<->SW 125GB/s alpha=0.35; inter-chassis 12.5GB/s alpha=2.6) ----
inline void BuildDGX2_TwoChassis(Topology& topo,
                                 bool allow_copy=false) {
  const int perChGpu = 16;
  topo.setNpusCount_(2*perChGpu);
  // two chassis NVSwitch blocks
  auto swA = topo.addSwitch("NVSW_A", allow_copy, perChGpu, perChGpu);
  auto swB = topo.addSwitch("NVSW_B", allow_copy, perChGpu, perChGpu);
  // core/aggregation switch (simplified to single aggregation layer, 12.5GB/s alpha=2.6us)
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/8, /*out*/8);

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
  // Inter-chassis via core: 12.5 GB/s α=2.6us (both directions)
  topo.addPhysLink(topo.switchNode(swA), topo.switchNode(core), 12.5, 2.60);
  topo.addPhysLink(topo.switchNode(core),topo.switchNode(swA),  12.5, 2.60);
  topo.addPhysLink(topo.switchNode(swB), topo.switchNode(core), 12.5, 2.60);
  topo.addPhysLink(topo.switchNode(core),topo.switchNode(swB),  12.5, 2.60);
  topo.finalizeReachability_();
}

// ---- NDv2: Four-chassis example (ref. TE-CCL Fig.14: intra-chassis 50/25GB/s alpha=0.7; inter-chassis 12.5GB/s alpha=1.3) ----
inline void BuildNDv2_FourChassis(Topology& topo,
                                  bool allow_copy=false) {
  const int ch = 4, perChGpu = 8;
  topo.setNpusCount_(ch*perChGpu);
  std::vector<SwitchID> tor(ch);
  for (int c = 0; c < ch; ++c) tor[c] = topo.addSwitch("ToR_"+std::to_string(c), allow_copy, perChGpu, perChGpu);
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch);
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
  topo.finalizeReachability_();
}

// ---- AMD MI250: 2/4 chassis (ref. TE-CCL Fig.15: 200/100/50/25GB/s alpha=0.6/0.75) ----
inline void BuildAMD_MI250_2Chassis(Topology& topo,
                                    bool allow_copy=false) {
  const int ch = 2, perChGpu = 16;
  topo.setNpusCount_(ch*perChGpu);
  // Leaf switch + aggregation switch per chassis (simplified)
  std::vector<SwitchID> leaf(ch), agg(ch);
  for (int c = 0; c < ch; ++c) {
    leaf[c] = topo.addSwitch("Leaf_"+std::to_string(c), allow_copy, perChGpu, perChGpu);
    agg[c]  = topo.addSwitch("Agg_"+std::to_string(c),  allow_copy, perChGpu, perChGpu);
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
  topo.finalizeReachability_();
}

inline void BuildAMD_MI250_4Chassis(Topology& topo, bool allow_copy=false) {
  // Extendable from 2ch: Agg[i] <-> Core 25GB/s, alpha=0.75, then Core interconnect
  const int ch = 4, perChGpu = 16;
  topo.setNpusCount_(ch*perChGpu);
  std::vector<SwitchID> leaf(ch), agg(ch);
  for (int c = 0; c < ch; ++c) {
    leaf[c] = topo.addSwitch("Leaf_"+std::to_string(c), allow_copy, perChGpu, perChGpu);
    agg[c]  = topo.addSwitch("Agg_"+std::to_string(c),  allow_copy, perChGpu, perChGpu);
    for (int g = 0; g < perChGpu; ++g) {
      double bw = (g<4)?200.0 : (g<8)?100.0 : 50.0;
      const int id = c*perChGpu + g;
      topo.addPhysLink(topo.deviceNode(id), topo.switchNode(leaf[c]), bw, 0.6);
      topo.addPhysLink(topo.switchNode(leaf[c]), topo.deviceNode(id), bw, 0.6);
    }
    topo.addPhysLink(topo.switchNode(leaf[c]), topo.switchNode(agg[c]), 25.0, 0.75);
    topo.addPhysLink(topo.switchNode(agg[c]),  topo.switchNode(leaf[c]), 25.0, 0.75);
  }
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch);
  for (int c = 0; c < ch; ++c) {
    topo.addPhysLink(topo.switchNode(agg[c]), topo.switchNode(core), 25.0, 0.75);
    topo.addPhysLink(topo.switchNode(core),   topo.switchNode(agg[c]), 25.0, 0.75);
  }
  topo.finalizeReachability_();
}

// NDv2 two-chassis (if needed)
inline void BuildNDv2_TwoChassis(Topology& topo, bool allow_copy=false) {
  const int ch = 2, perChGpu = 8;
  topo.setNpusCount_(ch*perChGpu);
  std::vector<SwitchID> tor(ch);
  for (int c = 0; c < ch; ++c) tor[c] = topo.addSwitch("ToR_"+std::to_string(c), allow_copy, perChGpu, perChGpu);
  // CORE
  auto core = topo.addSwitch("CORE", /*allow_copy*/false, /*in*/ch, /*out*/ch);
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
  topo.finalizeReachability_();
}

} // namespace tacos
