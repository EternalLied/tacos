# AllToAll 多路径负载均衡优化方案

## 当前问题分析

### 现有策略
1. **贪心单路径路由**: `findNextHopGreedy` 每次只选择一条最短路径
2. **随机选择**: 在多个等距离邻居中随机选一个
3. **后置条件动态调整**: 根据距离变化重新排序

### 主要瓶颈
1. **路径冲突严重**: 多个chunk竞争相同的中间节点和链路
2. **负载不均**: 某些热点链路过度使用,其他路径空闲
3. **无全局视野**: 贪心算法无法预见路径冲突

## 优化方案: 多路径负载均衡

### 方案1: 链路负载感知路由 (推荐实现)

**核心思想**: 在选择下一跳时,考虑链路的当前负载和未来预期负载

#### 实现步骤:

1. **添加链路负载追踪**
```cpp
// 在 TimeExpandedNetwork 中添加
std::unordered_map<std::pair<int,int>, int> linkPendingChunks_;  // 记录每条链路上等待传输的chunk数
std::unordered_map<std::pair<int,int>, double> linkUtilizationScore_;  // 链路繁忙程度评分
```

2. **修改 findNextHopGreedy 为 findNextHopLoadBalanced**
```cpp
int TimeExpandedNetwork::findNextHopLoadBalanced(
    const NpuID src, 
    const NpuID dest,
    const double loadBalancingWeight = 0.3  // 负载权重: 0-1之间
) const noexcept {
    
    // 收集所有能缩短距离的邻居
    std::vector<std::tuple<int, Time, double>> candidates;  // <neighbor, distance, load>
    
    for (const auto neighbor : directDeviceNeighbors_[src]) {
        Time neighborDist = getDistance(neighbor, dest);
        if (neighborDist >= currentDist) continue;
        
        // 计算该链路的负载评分
        auto linkKey = std::make_pair(src, neighbor);
        double loadScore = linkUtilizationScore_[linkKey];  // 0-1, 越小越好
        
        candidates.push_back({neighbor, neighborDist, loadScore});
    }
    
    // 综合评分: (1-w)*距离优先 + w*负载均衡
    auto bestCandidate = std::min_element(candidates.begin(), candidates.end(),
        [&](const auto& a, const auto& b) {
            double scoreA = (1.0 - loadBalancingWeight) * std::get<1>(a) / maxDist 
                          + loadBalancingWeight * std::get<2>(a);
            double scoreB = (1.0 - loadBalancingWeight) * std::get<1>(b) / maxDist
                          + loadBalancingWeight * std::get<2>(b);
            return scoreA < scoreB;
        });
    
    return std::get<0>(*bestCandidate);
}
```

3. **动态更新链路负载**
```cpp
// 在 transferChunkPartial 时更新
void TimeExpandedNetwork::transferChunkPartial(...) {
    auto linkKey = std::make_pair(src, dest);
    linkPendingChunks_[linkKey]++;
    updateLinkUtilizationScore(linkKey);
}

// 在 transferFinished 时更新
void TimeExpandedNetwork::transferFinished(...) {
    auto linkKey = std::make_pair(src, dest);
    linkPendingChunks_[linkKey]--;
    updateLinkUtilizationScore(linkKey);
}
```

**优势**:
- ✅ 实现简单,改动小
- ✅ 自动避开拥塞链路
- ✅ 保持贪心框架,不改变整体架构
- ✅ 可调节权重(0=纯贪心, 1=纯负载均衡)

**预期效果**: 提升20-40%性能

---

### 方案2: K路径分流策略

**核心思想**: 同一源到同一目的地的多个chunk,分散到K条不同的最短路径上

#### 实现步骤:

1. **预计算K条最短路径**
```cpp
// 使用Yen's K-shortest path算法
std::vector<std::vector<int>> findKShortestPaths(
    const NpuID src,
    const NpuID dest,
    const int K = 3
) const;
```

2. **轮询分配chunk到不同路径**
```cpp
// 在 Synthesizer 中维护路径分配表
std::unordered_map<std::pair<int,int>, int> pathAllocationCounter_;

int selectPathForChunk(const NpuID src, const NpuID dest) {
    auto key = std::make_pair(src, dest);
    int pathIndex = pathAllocationCounter_[key] % K;
    pathAllocationCounter_[key]++;
    return pathIndex;
}
```

3. **按选定路径调度**
```cpp
// 在 linkChunkMatching_ 中
int pathIndex = selectPathForChunk(originalSrc, dest);
const auto& path = kShortestPaths_[{originalSrc, dest}][pathIndex];
int nextHop = path[1];  // path[0]是src自己
```

**优势**:
- ✅ 显式分流,避免冲突
- ✅ 可预测性强
- ✅ 适合mesh等规则拓扑

**劣势**:
- ❌ 需要预计算K路径(内存开销)
- ❌ 不够动态,无法适应实时负载

**预期效果**: 提升30-50%性能

---

### 方案3: 虚拟流控制 (最优但复杂)

**核心思想**: 建模为流优化问题,使用线性规划求解全局最优路由

#### 实现步骤:

1. **建模为多商品流问题**
```
变量: f[c,e] = chunk c 在边 e 上的流量
目标: 最小化 max_time
约束:
  - 流守恒: ∑in = ∑out
  - 容量约束: ∑f[c,e] ≤ capacity[e]
  - 时间约束: time[e] = flow[e] / bandwidth[e]
```

2. **使用贪心近似**
```cpp
// 每N轮重新规划路由
if (currentIteration % replanInterval == 0) {
    auto flowSolution = solveMultiCommodityFlow();
    updateRoutingTable(flowSolution);
}
```

**优势**:
- ✅ 理论最优
- ✅ 全局视野

**劣势**:
- ❌ 实现复杂度高
- ❌ 计算开销大
- ❌ 与现有框架耦合度低

**预期效果**: 接近理论上限(~100-150 GiB/s)

---

## 推荐实施路线

### Phase 1: 链路负载感知 (2-3天)
1. 实现方案1的核心逻辑
2. 添加负载追踪机制
3. 调优权重参数(0.2-0.4)

### Phase 2: 测试与对比 (1天)
1. multi-round测试mesh2d
2. 对比原始性能
3. 分析链路利用率变化

### Phase 3: (可选) K路径分流 (3-5天)
1. 如果Phase 1效果不佳,实现方案2
2. 预计算2-3条路径
3. 轮询分配策略

## 预期性能提升

| 拓扑 | 当前 | 方案1 | 方案2 | 理论极限 |
|------|------|-------|-------|----------|
| mesh2d | 42 GiB/s | 55-60 GiB/s | 60-70 GiB/s | 114 GiB/s |
| 提升比例 | - | +30-43% | +43-66% | +171% |

## 关键参数调优

### 方案1参数:
- `loadBalancingWeight`: 0.2-0.4 (建议0.3)
- 更新频率: 每次chunk调度后立即更新
- 衰减因子: 历史负载按时间衰减

### 方案2参数:
- K值: 2-3条路径(mesh2d通常有2-4条等长路径)
- 分配策略: 轮询 vs 随机 vs 负载感知

## 代码改动位置

### 主要修改:
1. `time_expanded_network.h/cpp`: 添加负载追踪和新路由函数
2. `synthesizer.cpp`: 修改 `linkChunkMatching_` 调用新路由函数
3. 添加配置选项: 控制是否启用多路径优化

### 兼容性:
- 保持原有接口,通过开关控制
- 不影响AllGather等其他collective
