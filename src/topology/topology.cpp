/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <cassert>
#include <tacos/topology/topology.h>

using namespace tacos;

Topology::Topology() noexcept = default;

void Topology::setNpusCount_(const int npusCount) noexcept {
    assert(npusCount > 0);

    // set npusCount
    npusCount_ = npusCount;

    // allocate memory
    connected_ = decltype(connected_)(npusCount, std::vector<bool>(npusCount, false));
    latencies_ = decltype(latencies_)(npusCount, std::vector<Latency>(npusCount, 0));
    bandwidths_ = decltype(bandwidths_)(npusCount, std::vector<Bandwidth>(npusCount, 0));

    for (auto dest = 0; dest < npusCount; ++dest) {
        backtrackMap_[dest] = {};
    }

    // initialize switch-aware matrices
    viaSwitchId_.assign(npusCount_, std::vector<int>(npusCount_, -1));
    switchesCount_ = 0;
    switchMaxParallel_.clear();
}

Topology::Bandwidth Topology::bandwidth(NpuID src, NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(connected_[src][dest]);

    return bandwidths_[src][dest];
}

Topology::Latency Topology::latency(NpuID src, NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(connected_[src][dest]);

    return latencies_[src][dest];
}

std::vector<Topology::NpuID> Topology::backtrack(const NpuID dest) const noexcept {
    assert(0 <= dest && dest < npusCount_);
    assert(backtrackMap_.size() == npusCount_);

    return backtrackMap_.at(dest);
}

void Topology::connect_(const NpuID src,
                        const NpuID dest,
                        Bandwidth bandwidth,
                        Latency latency,
                        const bool bidirectional) noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // connect src -> dest
    connected_[src][dest] = true;
    bandwidths_[src][dest] = bandwidth;
    latencies_[src][dest] = latency;
    backtrackMap_[dest].push_back(src);

    if (bidirectional) {
        // connect dest -> src (if bi-directional)
        connect_(dest, src, bandwidth, latency, false);
    }
}

bool Topology::connected(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);

    return connected_[src][dest];
}

int Topology::npusCount() const noexcept {
    assert(npusCount_ > 0);

    return npusCount_;
}

// ===== Switch-aware (TE-CCL style) impl =====
SwitchID Topology::addSwitchUniform(const std::vector<NpuID>& ports,
                                    Bandwidth uplinkBandwidth,
                                    Latency gpu2swLatency,
                                    Latency sw2gpuLatency,
                                    int maxParallelEdges) noexcept {
    // Register switch
    const SwitchID sid = switchesCount_++;
    const int deg = static_cast<int>(ports.size());
    const int cap = (maxParallelEdges > 0) ? maxParallelEdges : deg; // TE-CCL Appendix C: min(in,out)=deg
    switchMaxParallel_.push_back(cap);

    // Create gpu->gpu synthetic links via this switch
    const Latency viaLatency = gpu2swLatency + sw2gpuLatency; // α_sw = α_g2s + α_s2g
    for (int i = 0; i < deg; ++i) {
        for (int j = 0; j < deg; ++j) {
            if (i == j) continue;
            const NpuID u = ports[i];
            const NpuID v = ports[j];
            // If there is already a direct edge u->v, keep it (prefer direct); otherwise add hyper-edge
            if (!connected_[u][v]) {
                connect_(u, v, uplinkBandwidth, viaLatency, false /*bi set below*/);
                connected_[u][v] = true;
                bandwidths_[u][v] = uplinkBandwidth;
                latencies_[u][v] = viaLatency;
                backtrackMap_[v].push_back(u);
                viaSwitchId_[u][v] = sid;
            }
        }
    }
    // make it bidirectional in a single pass (via connect_ above already adds both if true)
    for (int i = 0; i < deg; ++i) {
        for (int j = 0; j < deg; ++j) {
            if (i == j) continue;
            const NpuID u = ports[i];
            const NpuID v = ports[j];
            // mirror id for reverse if created above
            if (viaSwitchId_[u][v] == sid) {
                viaSwitchId_[v][u] = sid;
            }
        }
    }
    return sid;
}

int Topology::switchParallelLimit(const SwitchID sid) const noexcept {
    assert(sid >= 0 && sid < switchesCount_);
    return switchMaxParallel_[sid];
}

bool Topology::isViaSwitch(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    return viaSwitchId_[src][dest] >= 0;
}

int Topology::viaSwitchId(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    return viaSwitchId_[src][dest];
}
