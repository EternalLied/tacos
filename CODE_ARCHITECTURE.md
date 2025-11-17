# TACOS 项目代码架构与接口分析

## 项目概述

**TACOS** (Topology-Aware Collective Operation Synthesizer) 是一个用于优化集体通信模式的框架。它针对特定网络拓扑合成高效的集体通信算法，支持 AllGather、AllReduce 等集体操作。

**应用场景**：分布式机器学习、高性能计算中的进程间通信优化。

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                       main()                              │
│  - 定义网络拓扑（Mesh2D/Torus3D/Hypercube3D）            │
│  - 定义集体操作（AllGather）                             │
│  - 启动合成过程（Synthesizer.solve）                    │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
    ┌────────────┐   ┌─────────────┐   ┌──────────────┐
    │ Topology   │   │ Collective  │   │ Synthesizer  │
    │            │   │             │   │              │
    │ • Mesh2D   │   │ • AllGather │   │ • solve()    │
    │ • Torus3D  │   │ • AllReduce │   │ • TEN build  │
    │ • backtrack│   │ • chunks    │   │ • matching   │
    └────────────┘   └─────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                ┌─────────────────────────┐
                │   TimeExpandedNetwork   │
                │                         │
                │ • Link availability     │
                │ • Chunk routing         │
                │ • Time stepping         │
                └─────────────────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │    EventQueue/Timer     │
                │                         │
                │ • Schedule events       │
                │ • Track time            │
                └─────────────────────────┘
```

---

## 核心模块详解

### 1. Topology（拓扑模块）

#### 作用
定义网络中 NPU（Neural Processing Unit）之间的连接关系、带宽、延迟。

#### 核心类型定义
```cpp
using NpuID = int;              // NPU 编号
using Bandwidth = double;       // GiB/sec
using Latency = double;         // 微秒（us）
using SwitchID = int;          // 交换机编号（用于超边模型）
```

#### 核心数据结构

| 数据结构 | 用途 | 访问方式 |
|---------|------|---------|
| `connected_[src][dest]` | 记录两 NPU 是否连接 | bool |
| `bandwidths_[src][dest]` | 记录链路带宽 | GiB/sec |
| `latencies_[src][dest]` | 记录链路延迟 | 微秒 |
| `backtrackMap_[dest]` | 记录所有能向 dest 发送的源 | vector<NpuID> |

#### 核心接口

##### 查询接口
```cpp
// 获取两 NPU 之间的链路带宽
[[nodiscard]] Bandwidth bandwidth(NpuID src, NpuID dest) const noexcept;

// 获取两 NPU 之间的链路延迟
[[nodiscard]] Latency latency(NpuID src, NpuID dest) const noexcept;

// 【关键】反向追踪：找到所有能向目标 NPU 发送数据的源 NPU
// 这是合成过程中选择源的关键接口
[[nodiscard]] std::vector<NpuID> backtrack(NpuID dest) const noexcept;

// 获取网络中的 NPU 总数（不包括交换机）
[[nodiscard]] int npusCount() const noexcept;

// 检查两 NPU 是否连接
[[nodiscard]] bool connected(NpuID src, NpuID dest) const noexcept;
```

##### 构建接口（保护方法，供子类使用）
```cpp
// 设置网络中的 NPU 总数并初始化相关数据结构
void setNpusCount_(int npusCount) noexcept;

// 在两个 NPU 之间建立连接
// bidirectional=true 时自动建立反向链路
void connect_(NpuID src, NpuID dest, Bandwidth bandwidth, 
             Latency latency, bool bidirectional = false) noexcept;
```

##### 交换机相关接口
```cpp
// 注册一个交换机并为其连接的 GPU 创建合成超边
// 返回交换机 ID
SwitchID addSwitchUniform(const std::vector<NpuID>& ports,
                          Bandwidth uplinkBandwidth,
                          Latency gpu2swLatency,
                          Latency sw2gpuLatency,
                          int maxParallelEdges = -1) noexcept;

// 获取交换机的并发超边上限
[[nodiscard]] int switchParallelLimit(SwitchID sid) const noexcept;

// 检查链路是否通过交换机
[[nodiscard]] bool isViaSwitch(NpuID src, NpuID dest) const noexcept;
```

#### 具体实现类

**Mesh2D** - 二维网格拓扑
```cpp
Mesh2D(int width, int height, Bandwidth bw, Latency lat);
// 创建 width × height 的规则网格
// 网格中相邻 NPU 相连
```

**Torus2D/Torus3D** - 环面拓扑
- 二维或三维的环面网络
- 边界 NPU 环形连接

**Hypercube3D** - 三维超立方体
- 节点数为 2^3 = 8

#### `backtrack` 函数详解

##### 为什么需要？
集体通信合成过程中，需要为每个目标 NPU 选择最佳的源 NPU。给定目标，需要快速找出所有物理上能到达它的源。

##### 实现方式
```cpp
// 在 connect_ 中维护
backtrackMap_[dest].push_back(src);

// 查询时
std::vector<NpuID> Topology::backtrack(NpuID dest) const {
    return backtrackMap_.at(dest);  // O(1) 查询
}
```

##### 使用流程
```
1. 初始化阶段：build topology
   └─> connect(NPU0, NPU2, ...) 
   └─> connect(NPU1, NPU2, ...)
   └─> backtrackMap[2] = [0, 1]

2. 合成阶段：为 NPU2 选择源
   └─> sources = topology.backtrack(NPU2)  // [0, 1]
   └─> 遍历 sources，选择最优源
```

---

### 2. Collective（集体通信模式模块）

#### 作用
抽象描述集体通信模式，定义数据块的源-目标关系。

#### 核心概念

| 概念 | 含义 |
|------|------|
| **Chunk** | 数据块，集体操作的最小单位 |
| **ChunkID** | 数据块编号 |
| **ChunkSize** | 数据块大小（字节） |
| **Precondition** | 前置条件，数据块的源 NPU |
| **Postcondition** | 后置条件，数据块的目标 NPU 集合 |

#### 核心接口

```cpp
// 获取数据块的源 NPU
[[nodiscard]] NpuID precondition(ChunkID chunk) const noexcept;

// 获取数据块的目标 NPU 集合
[[nodiscard]] const std::unordered_set<NpuID>& postcondition(ChunkID chunk) const noexcept;

// 获取总数据块数
[[nodiscard]] int chunksCount() const noexcept;
```

#### AllGather 实现详解

##### 概念
全收集操作：每个 NPU 有一份数据，操作完成后每个 NPU 都拥有所有数据的副本。

##### 构造过程
```cpp
AllGather::AllGather(int npusCount, int collectivesCount) {
    // 假设：npusCount=4, collectivesCount=2
    
    // 创建目标集合（所有 NPU）
    dests = {0, 1, 2, 3}
    
    // 第一轮（collectivesCount=0）
    chunk(0): src=0, dest={0,1,2,3}
    chunk(1): src=1, dest={0,1,2,3}
    chunk(2): src=2, dest={0,1,2,3}
    chunk(3): src=3, dest={0,1,2,3}
    
    // 第二轮（collectivesCount=1）
    chunk(4): src=0, dest={0,1,2,3}
    chunk(5): src=1, dest={0,1,2,3}
    chunk(6): src=2, dest={0,1,2,3}
    chunk(7): src=3, dest={0,1,2,3}
    
    // 总数据块：npusCount * collectivesCount = 8
}
```

---

### 3. Synthesizer（合成器模块）

#### 作用
核心算法实现，根据拓扑和集体操作模式生成优化的通信方案。

#### 算法概述

**Link-Chunk Matching Algorithm**（链路-数据块匹配算法）

```
输入：
  - Topology：网络拓扑
  - Collective：集体操作模式
  - ChunkSize：每个数据块大小

输出：
  - collectiveTime：完成整个操作所需时间

过程：
  1. 初始化：
     - 构建 TimeExpandedNetwork
     - 标记所有源 NPU 已拥有其初始数据块
     
  2. 主循环（事件驱动）：
     while 还有未满足的数据块目标需求 do:
       a. 获取下一个时间事件
       b. 筛选当前时间所有未满足的后置条件
       c. 扩展时间扩展网络（TEN）
       d. 对每个未满足的(chunk, dest)对：
          - 通过 backtrack 找所有可能的源
          - 选择最优源（最早到达时间）
          - 安排数据块传输
```

#### 核心接口

```cpp
// 执行合成算法
// 返回完成整个集体操作所需的时间（微秒）
[[nodiscard]] Time solve(const Topology& topology,
                        const Collective& collective,
                        ChunkSize chunkSize) noexcept;
```

#### 核心数据结构

```cpp
// chunkMap_[chunk_id][npu_id]：是否在该 NPU 上
std::vector<std::vector<bool>> chunkMap_;

// 事件队列，管理时间推进
EventQueue eventQueue_;

// 当前模拟时间
Time currentTime_;

// 指向拓扑和集体操作的指针
const Topology* topology_;
const Collective* collective_;

// 时间扩展网络
std::unique_ptr<TimeExpandedNetwork> ten_;
```

#### 关键方法

```cpp
// 初始化：构建数据结构
void initialize_(const Topology& topology,
                 const Collective& collective,
                 ChunkSize chunkSize) noexcept;

// 标记初始数据块位置（源 NPU）
void markPrecondition_() noexcept;

// 筛选当前时间的未满足后置条件
PostconditionMap filterPostcondition_() const noexcept;

// 扩展时间扩展网络到当前时间步
void expandTenTimestep_(PostconditionMap* postconditionMap) noexcept;

// 为未满足的 (chunk, dest) 对执行链路-数据块匹配
void linkChunkMatching_(ChunkID chunk, NpuID dest) noexcept;
```

#### Link-Chunk Matching 详细过程

```cpp
void Synthesizer::linkChunkMatching_(ChunkID chunk, NpuID dest) {
    // 1. 反向追踪：找所有能向 dest 发送的源
    auto sources = ten_->backtrack(dest);
    
    // 2. 初始化最优候选
    auto bestTime = infinity;
    auto bestSources = vector<NpuID>();
    
    // 3. 遍历每个源
    for (const auto src : sources) {
        // 4. 过滤：源是否有该数据块？
        if (!chunkMap_[chunk][src]) continue;
        
        // 5. 评估：计算该路径的到达时间
        auto linkTransferTime = ten_->linkTransferTime(src, dest);
        auto arrivalTime = currentTime + linkTransferTime;
        
        // 6. 更新最优候选
        if (arrivalTime < bestTime) {
            bestTime = arrivalTime;
            bestSources = {src};
        } else if (arrivalTime == bestTime) {
            bestSources.push_back(src);
        }
    }
    
    // 7. 随机选择（若有多个最优源）
    auto selectedSrc = bestSources[random() % bestSources.size()];
    
    // 8. 安排传输
    ten_->schedule(src, dest, chunk, arrivalTime);
    
    // 9. 更新数据块位置
    chunkMap_[chunk][dest] = true;
    
    // 10. 调度到达事件
    eventQueue_.schedule(arrivalTime);
}
```

---

### 4. TimeExpandedNetwork（时间扩展网络）

#### 作用
模拟网络随时间的变化，追踪链路可用性和数据块位置。

#### 核心概念
将网络在不同时间步上"展开"，模拟数据块沿着网络的流动。

#### 核心接口

```cpp
// 获取当前时间步可用的源 NPU（已有数据块且链路可用）
[[nodiscard]] std::unordered_set<NpuID> backtrack(NpuID dest) noexcept;

// 计算从 src 到 dest 的传输时间
[[nodiscard]] Time linkTransferTime(NpuID src, NpuID dest) const noexcept;

// 检查链路在当前时间是否可用
[[nodiscard]] bool available(NpuID src, NpuID dest) const noexcept;

// 推进到下一个时间步
void timestep(Time time) noexcept;
```

#### 数据结构

```cpp
// available_[src][dest]：当前时间步该链路是否可用
std::vector<std::vector<bool>> available_;

// 当前时间
Time currentTime_;

// 参考的拓扑
const Topology& topology_;
```

---

### 5. EventQueue 和 Timer

#### EventQueue 的作用
事件驱动的时间管理，按时间顺序处理事件。

#### 核心接口

```cpp
// 调度一个事件（时间必须 > currentTime）
void schedule(Time time) noexcept;

// 取出下一个事件时间，并更新 currentTime
[[nodiscard]] Time pop() noexcept;

// 检查是否还有待处理事件
[[nodiscard]] bool empty() noexcept;

// 重置队列（初始时间为 0）
void reset() noexcept;
```

#### Timer 的作用
测量执行时间，用于性能监测。

```cpp
void start() noexcept;    // 开始计时
void stop() noexcept;     // 停止计时
[[nodiscard]] Time time() noexcept;  // 获取耗时（微秒）
```

---

## 完整执行流程

### main 函数执行步骤

```cpp
int main() {
    // 1. 创建网络拓扑
    //    定义网络的物理结构：NPU 数量、连接、带宽、延迟
    const auto topology = Mesh2D(width, height, bandwidth, latency);
    
    // 2. 创建集体操作
    //    定义数据块的源-目标关系
    const auto collective = AllGather(npusCount, collectivesCount);
    
    // 3. 创建合成器
    auto synthesizer = Synthesizer();
    
    // 4. 执行合成
    //    输入：拓扑、集体操作、数据块大小
    //    返回：预估完成时间
    auto collectiveTime = synthesizer.solve(
        topology,           // 网络拓扑
        collective,         // 集体操作
        chunkSize          // 每个数据块大小
    );
    
    // 5. 输出结果
    std::cout << "完成时间: " << collectiveTime << " 微秒" << std::endl;
    
    return 0;
}
```

### solve 函数详细过程

```
solve(topology, collective, chunkSize)
│
├─→ initialize_()
│   ├─ 重置事件队列
│   ├─ 构建时间扩展网络
│   └─ 初始化数据块位置矩阵
│
├─→ markPrecondition_()
│   └─ 标记所有源 NPU 已拥有其初始数据块
│
└─→ while eventQueue 非空:
    ├─ currentTime ← eventQueue.pop()
    ├─ postconditionMap ← filterPostcondition_()
    ├─ expandTenTimestep_()
    │  ├─ 更新 TEN 到当前时间步
    │  └─ 处理到达的数据块
    │
    └─ for each (chunk, dest) in postconditionMap:
       ├─ sources ← ten_->backtrack(dest)
       │            └─ 通过 Topology::backtrack 过滤后的源
       │
       ├─ 选择到达时间最早的源
       ├─ 安排传输事件
       ├─ 更新 chunkMap_[chunk][dest] = true
       └─ 调度到达事件到事件队列
```

---

## 关键设计模式

### 1. 策略模式（Strategy）
- `Topology` 作为基类
- 具体拓扑类（Mesh2D、Torus3D 等）实现不同策略

### 2. 模板方法模式（Template Method）
- `Collective` 定义集体操作框架
- 子类实现具体的块分配逻辑

### 3. 观察者模式（Observer）
- `EventQueue` 管理事件
- `Synthesizer` 响应事件进行处理

### 4. 构建者模式（Builder）
- `Topology::connect_` 逐步构建拓扑
- `Synthesizer::solve` 逐步合成通信方案

---

## 类型体系总结

```
Topology（抽象基类）
├─ Mesh2D
├─ Mesh2DHetero
├─ Torus2D
├─ Torus3D
└─ Hypercube3D

Collective（抽象基类）
└─ AllGather

Synthesizer
├─ uses TimeExpandedNetwork
├─ uses EventQueue
└─ uses Timer

TimeExpandedNetwork
└─ references Topology

EventQueue
└─ uses std::priority_queue<Time>
```

---

## 性能考量

### 时间复杂度分析

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `Topology::bandwidth` | O(1) | 直接矩阵访问 |
| `Topology::backtrack` | O(1) | 哈希表查询 |
| `EventQueue::schedule` | O(log E) | E = 事件数 |
| `EventQueue::pop` | O(log E) | 堆操作 |
| `Synthesizer::solve` | O(T × C × D²) | T=时间步，C=块数，D=度数 |

### 空间复杂度

```
Topology:
  - connected_:    O(N²)    (N = NPU 数)
  - bandwidths_:   O(N²)
  - latencies_:    O(N²)
  - backtrackMap_: O(N·E)   (E = 边数)

Synthesizer:
  - chunkMap_:     O(C·N)   (C = 块数, N = NPU 数)
  - eventQueue_:   O(E)     (E = 事件数)
```

---

## 扩展点

1. **新的拓扑类型**
   - 继承 `Topology`，实现 `setNpusCount_` 和 `connect_`

2. **新的集体操作**
   - 继承 `Collective`，实现 `precondition` 和 `postcondition`

3. **自定义匹配策略**
   - 修改 `Synthesizer::linkChunkMatching_` 中的源选择逻辑

4. **链路约束**
   - 在 `TimeExpandedNetwork::available` 中添加额外约束

---

## 参考链接

- **论文**：[TACOS: Topology-Aware Collective Operation Synthesizer](https://arxiv.org/abs/2310.16389)
- **代码库**：[GitHub - TACOS](https://github.com/EternalLied/tacos)

