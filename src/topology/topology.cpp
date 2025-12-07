/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <cassert>
#include <tacos/topology/topology.h>
#include <queue>
#include <limits>

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

    // reset physical graph (switches/links can be re-added)
    switches_.clear();
    switchesCount_ = 0;
    physLinks_.clear();

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

// ===== multi-switch additions =====
SwitchID Topology::addSwitch(const std::string& name, bool allowCopy,
                                       int inCap, int outCap, 
                                       SwitchForwardingMode mode) noexcept {
    Switch sw;
    sw.name = name;
    sw.allowCopy = allowCopy;
    sw.inCap = inCap;
    sw.outCap = outCap;
    sw.forwardingMode = mode;
    switches_.push_back(sw);
    return switchesCount_++;
}

void Topology::addPhysLink(const NodeIndex u, const NodeIndex v,
                           const Bandwidth bw, const Latency alpha) noexcept {
    physLinks_.push_back(PhysLink{u, v, bw, alpha});
}

int Topology::switchInCapResolved(const SwitchID sid) const noexcept {
    const auto& sw = switches_[sid];
    if (sw.inCap > 0) return sw.inCap;
    // default: degree from incoming physLinks
    int deg = 0;
    for (const auto& e : physLinks_) if (e.dst == switchNode(sid)) ++deg;
    return std::max(1, deg);
}

int Topology::switchOutCapResolved(const SwitchID sid) const noexcept {
    const auto& sw = switches_[sid];
    if (sw.outCap > 0) return sw.outCap;
    int deg = 0;
    for (const auto& e : physLinks_) if (e.src == switchNode(sid)) ++deg;
    return std::max(1, deg);
}

// Get link statistics by type
std::tuple<int, int, int, int> Topology::getLinkStatistics() const noexcept {
    int deviceToDevice = 0;
    int deviceToSwitch = 0;
    int switchToDevice = 0;
    int switchToSwitch = 0;

    for (const auto& link : physLinks_) {
        const bool srcIsDevice = (link.src < npusCount_);
        const bool dstIsDevice = (link.dst < npusCount_);

        if (srcIsDevice && dstIsDevice) {
            ++deviceToDevice;
        } else if (srcIsDevice && !dstIsDevice) {
            ++deviceToSwitch;
        } else if (!srcIsDevice && dstIsDevice) {
            ++switchToDevice;
        } else {
            ++switchToSwitch;
        }
    }

    return std::make_tuple(deviceToDevice, deviceToSwitch, switchToDevice, switchToSwitch);
}

// Compute GPU->GPU reachability from the physical graph
void Topology::finalizeReachability_() noexcept {
    // init
    for (int u = 0; u < npusCount_; ++u) {
        for (int v = 0; v < npusCount_; ++v) {
            connected_[u][v] = (u == v ? false : connected_[u][v]);
            bandwidths_[u][v] = bandwidths_[u][v];
            latencies_[u][v] = latencies_[u][v];
        }
    }
    // BFS/Dijkstra over physical graph to check reachability
    const int T = totalNodes();
    std::vector<std::vector<int>> adj(T);
    for (int i = 0; i < (int)physLinks_.size(); ++i) {
        adj[physLinks_[i].src].push_back(i); // store edge index
    }
    for (int s = 0; s < npusCount_; ++s) {
        std::vector<char> seen(T, 0);
        std::queue<int> q;
        seen[s] = 1; q.push(s);
        while(!q.empty()) {
            int x = q.front(); q.pop();
            for (int ei : adj[x]) {
                const auto& e = physLinks_[ei];
                if (!seen[e.dst]) { seen[e.dst] = 1; q.push(e.dst); }
            }
        }
        for (int t = 0; t < npusCount_; ++t) {
            if (s == t) continue;
            if (seen[t]) {
                connected_[s][t] = true;
                // bandwidth/latency at GPU-level is only for "estimation"; actual scheduling is calculated by TEN along the path
                // Keep 0 here; TEN's shortest path calculation will fill in the actual end-to-end time
            }
        }
    }
}
