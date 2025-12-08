# 重构提案：移除逻辑链路抽象，统一使用物理边描述

## 背景

当前 TimeExpandedNetwork 维护了两套并行的状态：

1. **逻辑链路状态**（GPU-to-GPU）
   - `available_[src][dest]` - 是否可用
   - `linkBusyUntil_[src][dest]` - 繁忙到何时
   - `chunk_[src][dest]` - 传输的块 ID

2. **物理边状态**（真实资源）
   - `edgeBusyUntil_[u][v]` - 物理边繁忙时间
   - `routes_[src][dest]` - GPU 间的物理路径
   - `swInUse_/swOutUse_` - 交换机端口占用

## 问题

### 1. 状态冗余和不一致风险
```cpp
// 当前代码同时更新两套状态：
available_[src][dest] = false;           // 逻辑状态
reserveRoute_(routes_[src][dest], t);    // 物理状态

// 如果中间某个步骤失败，两套状态就不一致了
```

### 2. 逻辑链路抽象在交换机拓扑中语义不清
- `available_[0][1]` 的真实含义是什么？
  - 路径 `0→8→1` 的所有物理边都空闲？
  - 交换机 8 有足够的端口容量？
  - 实际上这只是一个**缓存的快照**，可能已经过时

### 3. 实际决策已经完全基于物理边
```cpp
bool TimeExpandedNetwork::canReserveRoute_(const Route& r, const Time t0) {
    // 检查的是 edgeBusyUntil_[u][v]，不是 available_[src][dest]
    if (edgeBusyUntil_[u][v] > t) return false;
    if (!swCapOkAt_(switchId, t, t + d, isIn)) return false;
}
```

## 重构方案

### 步骤 1：修改 `available()` 基于物理路径实时检查

**修改前**：
```cpp
bool TimeExpandedNetwork::available(NpuID src, NpuID dest) const noexcept {
    return available_[src][dest];  // 返回缓存的状态
}
```

**修改后**：
```cpp
bool TimeExpandedNetwork::available(NpuID src, NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    
    if (src == dest) return true;
    
    // 直接检查物理路径是否可预留
    const auto& route = routes_[src][dest];
    if (route.nodes.empty()) return false;
    
    return canReserveRoute_(route, currentTime_);
}
```

### 步骤 2：修改 `chunk()` 基于物理边查询

**修改前**：
```cpp
ChunkID TimeExpandedNetwork::chunk(NpuID src, NpuID dest) const noexcept {
    return chunk_[src][dest];  // 从逻辑链路数组读取
}
```

**修改后**：
```cpp
// 方案 A：在 Route 中存储 chunk ID
struct Route {
    std::vector<int> nodes;
    std::vector<Time> deltas;
    int logicalHops = 0;
    int physicalHops = 0;
    ChunkID currentChunk = -1;  // ✨ 新增：当前传输的块
    Time busyUntil = -1;        // ✨ 新增：繁忙到何时
};

ChunkID TimeExpandedNetwork::chunk(NpuID src, NpuID dest) const noexcept {
    const auto& route = routes_[src][dest];
    return route.currentChunk;
}
```

**或方案 B：维护 GPU-pair 到 chunk 的映射**
```cpp
// 使用 map 而不是二维数组，节省内存
std::map<std::pair<NpuID, NpuID>, ChunkID> activeChunks_;

ChunkID TimeExpandedNetwork::chunk(NpuID src, NpuID dest) const noexcept {
    auto it = activeChunks_.find({src, dest});
    return (it != activeChunks_.end()) ? it->second : -1;
}
```

### 步骤 3：修改 `transferChunk()` 只更新物理资源

**修改前**：
```cpp
void TimeExpandedNetwork::transferChunk(NpuID src, NpuID dest, ChunkID chunk, Time time) {
    // 更新逻辑链路状态
    available_[src][dest] = false;
    chunk_[src][dest] = chunk;
    linkBusyUntil_[src][dest] = time;
    
    // 更新物理边状态
    reserveRoute_(routes_[src][dest], currentTime_);
}
```

**修改后**：
```cpp
void TimeExpandedNetwork::transferChunk(NpuID src, NpuID dest, ChunkID chunk, Time time) {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(chunk >= 0);
    assert(time >= currentTime_);
    
    auto& route = routes_[src][dest];
    assert(!route.nodes.empty());
    
    // 只更新物理资源（一处真相）
    reserveRoute_(route, currentTime_);
    
    // 记录传输信息到路由（或使用 map）
    route.currentChunk = chunk;
    route.busyUntil = time;
}
```

### 步骤 4：修改 `transferFinished()` 清理传输状态

**修改前**：
```cpp
void TimeExpandedNetwork::transferFinished(NpuID src, NpuID dest) {
    available_[src][dest] = true;
    linkBusyUntil_[src][dest] = -1;
    chunk_[src][dest] = -1;
}
```

**修改后**：
```cpp
void TimeExpandedNetwork::transferFinished(NpuID src, NpuID dest) {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    
    auto& route = routes_[src][dest];
    route.currentChunk = -1;
    route.busyUntil = -1;
    
    // 物理边的 edgeBusyUntil_ 会随时间推移自然过期，无需手动清理
}
```

### 步骤 5：移除 `timestep()` 中的逻辑链路重置

**修改前**：
```cpp
void TimeExpandedNetwork::timestep(Time time) {
    currentTime_ = time;
    
    // 重置所有 GPU 对的逻辑链路可用性
    for (int src = 0; src < npusCount_; ++src) {
        for (int dest = 0; dest < npusCount_; ++dest) {
            if (linkBusyUntil_[src][dest] > currentTime_) {
                available_[src][dest] = false;
            } else {
                available_[src][dest] = topology_.connected(src, dest);
            }
        }
    }
}
```

**修改后**：
```cpp
void TimeExpandedNetwork::timestep(Time time) {
    assert(time > currentTime_);
    currentTime_ = time;
    
    // 无需重置逻辑状态，available() 会实时计算
    // 物理边状态会在查询时自然过期（edgeBusyUntil_[u][v] < currentTime_）
}
```

### 步骤 6：从头文件移除数据成员

**修改 `time_expanded_network.h`**：
```cpp
// 移除这三个成员变量：
// std::vector<std::vector<bool>> available_;       // ❌ 删除
// std::vector<std::vector<Time>> linkBusyUntil_;   // ❌ 删除
// std::vector<std::vector<ChunkID>> chunk_;        // ❌ 删除

// Route 结构体新增字段：
struct Route {
    std::vector<int> nodes;
    std::vector<Time> deltas;
    int logicalHops = 0;
    int physicalHops = 0;
    ChunkID currentChunk = -1;  // ✨ 新增
    Time busyUntil = -1;        // ✨ 新增
};
```

## 优势

### ✅ 1. 单一真相来源（Single Source of Truth）
- 物理边状态 `edgeBusyUntil_[u][v]` 是唯一的资源占用记录
- 交换机容量 `swInUse_/swOutUse_` 是唯一的端口占用记录
- 不再有"逻辑说可用，物理却冲突"的风险

### ✅ 2. 更精确的可用性检查
```cpp
// 旧方式：读取缓存（可能过时）
if (available_[src][dest]) { ... }

// 新方式：实时检查物理路径（总是正确）
if (canReserveRoute_(routes_[src][dest], currentTime_)) { ... }
```

### ✅ 3. 减少内存占用
- 移除三个 O(N²) 的二维数组（N = GPU 数量）
- 对于 8 GPU 系统：节省 3 × 64 = 192 字节（小系统）
- 对于 1024 GPU 系统：节省 3 × 1M = 3MB（大系统）

### ✅ 4. 代码更清晰
- `transferChunk()` 只需调用 `reserveRoute_()`
- `available()` 就是 `canReserveRoute()` 的语法糖
- 逻辑更直观，维护更简单

### ✅ 5. 可扩展性更好
- 新增交换机类型（如 Store-and-Forward）无需修改逻辑层
- 新增路由策略（如动态路由）只需修改物理层
- 上层 API 保持不变

## 潜在风险与应对

### ⚠️ 风险 1：性能开销
**问题**：每次 `available()` 都调用 `canReserveRoute_()`，遍历整个路径

**应对**：
1. **缓存短期结果**：在同一 timestep 内缓存查询结果
2. **批量检查**：调度器批量检查多个 GPU 对，避免重复计算
3. **性能测试**：对比重构前后的 benchmark

### ⚠️ 风险 2：API 语义变化
**问题**：`available()` 从"读状态"变为"计算结果"

**应对**：
- API 签名不变，调用者无感知
- 内部实现从 O(1) 变为 O(hops)，但路径通常很短（2-3 跳）
- 如果性能敏感，可添加注释说明

### ⚠️ 风险 3：现有测试用例可能失败
**问题**：测试可能依赖 `available_[]` 的具体行为

**应对**：
- 逐步重构，先添加新方法，再替换旧方法
- 保留旧数据结构作为"影子"进行对比验证
- 确认所有测试通过后再删除

## 实施计划

### Phase 1: 准备阶段（不破坏现有代码）
1. ✅ 在 `Route` 中添加 `currentChunk` 和 `busyUntil` 字段
2. ✅ 添加新方法 `availableV2()` 基于物理路径检查
3. ✅ 添加新方法 `chunkV2()` 从 Route 读取
4. ✅ 添加日志对比新旧方法的结果差异

### Phase 2: 验证阶段
1. ✅ 运行所有测试用例，确认新方法结果一致
2. ✅ 性能 benchmark：测量 `availableV2()` 开销
3. ✅ 如果性能下降 > 10%，添加缓存层

### Phase 3: 切换阶段
1. ✅ 修改 `available()` 调用 `availableV2()` 实现
2. ✅ 修改 `chunk()` 调用 `chunkV2()` 实现
3. ✅ 修改 `transferChunk()` 和 `transferFinished()`
4. ✅ 运行测试确认无回归

### Phase 4: 清理阶段
1. ✅ 移除 `available_`, `linkBusyUntil_`, `chunk_` 数据成员
2. ✅ 移除构造函数中的初始化代码
3. ✅ 移除 `timestep()` 中的重置逻辑
4. ✅ 更新文档和注释

## 结论

**强烈建议执行此重构**，原因：

1. **正确性** > 性能：单一真相来源避免状态不一致
2. **可维护性**：代码逻辑更清晰，物理资源管理统一
3. **可扩展性**：支持更复杂的拓扑和路由策略
4. **内存效率**：移除冗余数据结构

交换机拓扑已经打破了"逻辑链路 = 物理边"的简单假设，继续维护两套状态不仅增加复杂度，还埋下了一致性隐患。现在正是重构的最佳时机。

---

**下一步**：我可以立即实施 Phase 1 的代码修改，或者您想先讨论其他方案？
