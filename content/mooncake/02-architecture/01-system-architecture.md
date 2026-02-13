---
title: "整体架构设计"
weight: 1
---

[上一篇：核心概念与关键技术](../01-overview/02-core-concepts.md) | [目录](../README.md) | [下一篇：请求处理流程详解](02-request-lifecycle.md)

---

# 整体架构设计

### 3.1 架构总览

Mooncake 采用了一个精心设计的分层架构，将 LLM 推理系统分解为多个独立但协作的组件。这种设计使得系统能够在保持高性能的同时，实现灵活的资源管理和调度。

#### 3.1.1 核心组件

```mermaid
graph TB
    subgraph "控制平面 (Control Plane)"
        Conductor[Conductor<br/>全局调度器]
        MetaStore[Metadata Store<br/>元数据存储]
    end

    subgraph "数据平面 (Data Plane)"
        subgraph "Prefill 节点池"
            P1[Prefill Node 1]
            P2[Prefill Node 2]
            Pn[Prefill Node N]
        end

        subgraph "Decode 节点池"
            D1[Decode Node 1]
            D2[Decode Node 2]
            Dn[Decode Node N]
        end

        subgraph "Mooncake Store (分布式 KVCache)"
            MS1[Store Node 1<br/>CPU/GPU/SSD]
            MS2[Store Node 2<br/>CPU/GPU/SSD]
            MSn[Store Node N<br/>CPU/GPU/SSD]
        end
    end

    subgraph "传输层 (Transfer Layer)"
        TE[Transfer Engine<br/>RDMA/TCP/NVMe-oF]
    end

    Client[客户端请求] --> Conductor
    Conductor --> P1 & P2 & Pn
    Conductor --> D1 & D2 & Dn
    P1 & P2 & Pn <--> TE
    D1 & D2 & Dn <--> TE
    TE <--> MS1 & MS2 & MSn
    Conductor <--> MetaStore
```

Mooncake 系统由以下核心组件构成：

| 组件 | 职责 | 关键特性 |
|------|------|----------|
| **Conductor** | 全局调度器 | Cache-aware 调度、负载均衡、SLO 保证 |
| **Prefill Nodes** | 处理输入 tokens | 计算密集型、生成 KVCache |
| **Decode Nodes** | 生成输出 tokens | 内存密集型、消费 KVCache |
| **Mooncake Store** | 分布式 KVCache 存储 | 多级存储、跨节点共享 |
| **Transfer Engine** | 高性能数据传输 | RDMA、拓扑感知、多协议支持 |
| **Metadata Store** | 元数据管理 | 基于 etcd、高可用 |

#### 3.1.2 代码模块对应关系

Mooncake 的代码结构与上述架构紧密对应：

```
Mooncake/
├── mooncake-transfer-engine/    # Transfer Engine 核心实现
│   ├── include/
│   │   ├── transfer_engine.h    # 主入口 API
│   │   ├── transport/           # 各种传输协议实现
│   │   │   ├── rdma_transport/  # RDMA 传输
│   │   │   ├── tcp_transport/   # TCP 传输
│   │   │   └── nvmeof_transport/# NVMe-oF 传输
│   │   ├── topology.h           # 拓扑发现
│   │   └── transfer_metadata.h  # 元数据管理
│   └── src/
│       ├── transfer_engine.cpp  # Transfer Engine 实现
│       └── transport/           # 传输层实现
│
├── mooncake-store/              # Mooncake Store 实现
│   ├── include/
│   │   ├── storage_backend.h    # 存储后端抽象
│   │   ├── eviction_strategy.h  # 缓存驱逐策略
│   │   ├── master_service.h     # Master 服务
│   │   └── ha_helper.h          # 高可用支持
│   └── src/
│       ├── ha_helper.cpp        # Leader 选举实现
│       └── master_service.cpp   # Master 服务实现
│
├── mooncake-integration/        # 与推理框架的集成
│   ├── vllm/                    # vLLM 集成
│   └── sglang/                  # SGLang 集成
│
└── mooncake-wheel/              # Python 包装器和 CLI 工具
```

### 3.2 P/D 分离架构详解

#### 3.2.1 分离架构的核心思想

P/D（Prefill/Decode）分离是 Mooncake 架构的核心设计决策。与传统的将两个阶段耦合在同一节点的方案不同，Mooncake 将它们部署在独立的节点池中：

```mermaid
graph TB
    subgraph "传统耦合架构"
        direction LR
        TR[请求] --> TN1[节点 1]
        TR --> TN2[节点 2]
        TN1 --> TP1[Prefill] --> TD1[Decode]
        TN2 --> TP2[Prefill] --> TD2[Decode]
        style TP1 fill:#ff9999
        style TD1 fill:#99ccff
        style TP2 fill:#ff9999
        style TD2 fill:#99ccff
    end

    subgraph "Mooncake P/D 分离架构"
        direction LR
        MR[请求] --> MC[Conductor]
        MC --> MP1[Prefill 1]
        MC --> MP2[Prefill 2]
        MP1 --> MKV[KVCache<br/>传输]
        MP2 --> MKV
        MKV --> MD1[Decode 1]
        MKV --> MD2[Decode 2]
        style MP1 fill:#ff9999
        style MP2 fill:#ff9999
        style MD1 fill:#99ccff
        style MD2 fill:#99ccff
    end
```

#### 3.2.2 分离带来的优势

**1. 消除阶段干扰**

在耦合架构中，长 Prefill 请求会阻塞同节点上的 Decode 请求，导致 TBT 超时。分离后：
- Prefill 节点可以专注于处理计算密集型任务
- Decode 节点不受 Prefill 干扰，TBT 稳定可控

**2. 独立资源配置**

| 阶段 | 资源需求 | 优化方向 |
|------|----------|----------|
| Prefill | 高计算能力、中等内存 | 最大化 MFU |
| Decode | 高内存带宽、大 batch size | 最大化并发请求数 |

分离后可以针对不同阶段优化配置：
- Prefill 节点可以使用计算更强的 GPU
- Decode 节点可以配置更大的显存和 batch size

**3. 弹性伸缩**

可以根据工作负载特征独立调整 P/D 节点比例：
- 长上下文场景：增加 Prefill 节点
- 高并发场景：增加 Decode 节点

#### 3.2.3 分离带来的挑战：KVCache 传输

P/D 分离的代价是需要在 Prefill 完成后将 KVCache 传输到 Decode 节点。这引入了额外的网络开销：

**传输量估算**（以 LLaMA3-70B 为例）：

| 上下文长度 | KVCache 大小 | 100GB/s 网络传输时间 |
|------------|--------------|---------------------|
| 4K tokens | 1.28 GB | ~13 ms |
| 32K tokens | 10.24 GB | ~100 ms |
| 128K tokens | 40.96 GB | ~400 ms |

**关键洞察**：Mooncake 通过以下机制将传输开销最小化：

1. **Prefix Caching**：如果目标 Decode 节点已有部分 KVCache 缓存，只需传输差异部分
2. **并行传输**：利用多网卡聚合带宽
3. **流水线**：KVCache 可以边生成边传输

### 3.3 Mooncake Store 设计概览

Mooncake Store 是系统的核心组件，实现了分布式 KVCache 存储池。

#### 3.3.1 存储层次结构

```mermaid
graph TB
    subgraph "存储层次 (从快到慢)"
        L1[GPU VRAM<br/>~2 TB/s 带宽]
        L2[CPU DRAM<br/>~400 GB/s 带宽]
        L3[NVMe SSD<br/>~10-50 GB/s 带宽]
    end

    L1 --> L2 --> L3

    subgraph "容量 vs 速度"
        C1[少量高热点数据]
        C2[中等热度数据]
        C3[大量冷数据]
    end

    L1 -.-> C1
    L2 -.-> C2
    L3 -.-> C3
```

Mooncake Store 利用集群中所有节点的多级存储资源：

| 存储层 | 介质 | 典型带宽 | 典型容量 (8 节点) |
|--------|------|----------|------------------|
| L1 | GPU VRAM | 2-3 TB/s | 640 GB - 1.28 TB |
| L2 | CPU DRAM | 400 GB/s | 8-16 TB |
| L3 | NVMe SSD | 10-50 GB/s | 数十 TB |

#### 3.3.2 KVCache 对象模型

Mooncake Store 将 KVCache 组织为树形结构：

```mermaid
graph TB
    Root[根节点<br/>空前缀] --> A[Token 序列 A]
    Root --> B[Token 序列 B]
    A --> A1[A 的扩展 1]
    A --> A2[A 的扩展 2]
    A1 --> A1a[A1 的扩展]
    B --> B1[B 的扩展 1]

    style Root fill:#f0f0f0
    style A fill:#ffe0e0
    style B fill:#e0ffe0
    style A1 fill:#ffe0e0
    style A2 fill:#ffe0e0
    style A1a fill:#ffe0e0
    style B1 fill:#e0ffe0
```

**Radix Tree 结构特点**：
- 公共前缀共享存储空间
- 支持快速前缀匹配
- 天然支持 Prefix Caching

#### 3.3.3 分片与复制策略

KVCache 被分成固定大小的 **Chunk**（通常 4KB-64KB），每个 Chunk 可以：
- 存储在不同节点上
- 拥有多个副本（用于热点数据）
- 在不同存储层之间迁移

```mermaid
graph LR
    subgraph "KVCache 对象"
        K[KVCache<br/>128K tokens]
        K --> C1[Chunk 1]
        K --> C2[Chunk 2]
        K --> C3[Chunk N]
    end

    subgraph "分布式存储"
        N1[节点 1]
        N2[节点 2]
        N3[节点 3]
    end

    C1 --> N1
    C1 -.副本.-> N2
    C2 --> N2
    C3 --> N3
    C3 -.副本.-> N1
```

### 3.4 Transfer Engine 设计概览

Transfer Engine 是 Mooncake 的高性能数据传输层，负责在节点间高效传输 KVCache 数据。

#### 3.4.1 核心设计目标

1. **高带宽**：充分利用多 RDMA 网卡的聚合带宽
2. **低延迟**：最小化传输开销
3. **拓扑感知**：自动选择最优传输路径
4. **协议抽象**：统一 API 支持多种传输协议

#### 3.4.2 架构分层

```mermaid
graph TB
    subgraph "应用层"
        App[LLM 推理引擎<br/>vLLM / SGLang]
    end

    subgraph "Transfer Engine API"
        TE[TransferEngine]
        TE --> AM[allocateBatchID]
        TE --> ST[submitTransfer]
        TE --> GS[getTransferStatus]
    end

    subgraph "Transport Layer"
        MT[MultiTransport]
        MT --> RDMA[RdmaTransport]
        MT --> TCP[TcpTransport]
        MT --> NVME[NvmeofTransport]
    end

    subgraph "Hardware Abstraction"
        RDMA --> RC[RdmaContext]
        RDMA --> RE[RdmaEndpoint]
        RC --> NIC1[NIC 1]
        RC --> NIC2[NIC 2]
    end

    App --> TE
    TE --> MT
```

#### 3.4.3 关键代码结构

从 `transfer_engine.h` 中可以看到 Transfer Engine 的核心 API：

```cpp
class TransferEngine {
public:
    // 初始化引擎
    int init(const std::string& metadata_conn_string,
             const std::string& local_server_name,
             const std::string& ip_or_host_name,
             uint64_t rpc_port);

    // 安装传输协议
    Transport* installTransport(const std::string& proto, void** args);

    // 注册本地内存区域 (用于 RDMA)
    int registerLocalMemory(void* addr, size_t length,
                           const std::string& location,
                           bool remote_accessible,
                           bool update_metadata = true);

    // 分配 Batch ID (用于批量传输)
    BatchID allocateBatchID(size_t batch_size);

    // 提交传输请求
    Status submitTransfer(BatchID batch_id,
                         const std::vector<TransferRequest>& entries);

    // 获取传输状态
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                            TransferStatus& status);

    // 打开远程 Segment
    SegmentHandle openSegment(const std::string& segment_name);
};
```

**核心数据结构**（来自 `transport.h`）：

```cpp
struct TransferRequest {
    enum OpCode { READ, WRITE };

    OpCode opcode;           // 读或写操作
    void* source;            // 本地地址
    SegmentID target_id;     // 目标 Segment
    uint64_t target_offset;  // 目标偏移
    size_t length;           // 传输长度
};

enum TransferStatusEnum {
    WAITING,     // 等待执行
    PENDING,     // 执行中
    COMPLETED,   // 完成
    FAILED,      // 失败
    TIMEOUT      // 超时
};
```

### 3.5 Conductor 调度器概览

Conductor 是 Mooncake 的全局调度器，负责将请求分配到合适的 Prefill 和 Decode 节点。

#### 3.5.1 调度目标

Conductor 需要同时优化多个目标：

```mermaid
graph LR
    subgraph "调度目标"
        G1[最小化 TTFT]
        G2[满足 TBT SLO]
        G3[最大化 Cache 命中]
        G4[负载均衡]
    end

    subgraph "约束条件"
        C1[节点容量]
        C2[网络带宽]
        C3[SLO 阈值]
    end

    G1 & G2 & G3 & G4 --> D[调度决策]
    C1 & C2 & C3 --> D
```

#### 3.5.2 Cache-aware 调度

Conductor 的核心创新是 **Cache-aware** 调度——将请求调度到能够复用最多 KVCache 的节点：

```mermaid
sequenceDiagram
    participant R as 新请求
    participant C as Conductor
    participant MS as Mooncake Store
    participant P as Prefill Nodes

    R->>C: 请求 (prefix tokens)
    C->>MS: 查询 prefix 位置
    MS-->>C: 返回各节点缓存情况
    C->>C: 计算最优节点<br/>(最大缓存复用 + 负载均衡)
    C->>P: 调度到最优 Prefill 节点
```

**调度算法核心思想**：

给定请求 $r$ 和候选节点集合 $N$，调度得分为：

$$score(r, n) = \alpha \cdot \frac{cached\_prefix\_length(r, n)}{total\_prefix\_length(r)} + \beta \cdot (1 - load(n))$$

其中：
- $cached\_prefix\_length(r, n)$：节点 $n$ 上已缓存的请求 $r$ 的前缀长度
- $total\_prefix\_length(r)$：请求 $r$ 的总前缀长度
- $load(n)$：节点 $n$ 的当前负载
- $\alpha, \beta$：权重参数

### 3.6 元数据管理

Mooncake 使用 etcd 作为分布式元数据存储，管理以下信息：

#### 3.6.1 元数据类型

| 元数据类型 | 内容 | 更新频率 |
|------------|------|----------|
| Segment 信息 | 节点地址、端口、协议 | 节点启动/退出时 |
| Buffer 描述 | 内存地址、大小、rkey | 内存注册时 |
| 拓扑信息 | NIC 配置、存储层次 | 节点启动时 |
| Master 视图 | 当前 Master 地址 | Leader 选举时 |

#### 3.6.2 高可用设计

从 `ha_helper.cpp` 可以看到 Master 选举机制：

```cpp
void MasterViewHelper::ElectLeader(const std::string& master_address,
                                   ViewVersionId& version,
                                   EtcdLeaseId& lease_id) {
    while (true) {
        // 1. 检查是否已有 Leader
        auto ret = EtcdHelper::Get(master_view_key_, ...);

        if (ret != ErrorCode::ETCD_KEY_NOT_EXIST) {
            // 已有 Leader，等待其失效
            EtcdHelper::WatchUntilDeleted(master_view_key_, ...);
            continue;
        }

        // 2. 尝试成为 Leader
        // 获取租约
        EtcdHelper::GrantLease(ETCD_MASTER_VIEW_LEASE_TTL, lease_id);

        // 原子创建 key (带租约)
        ret = EtcdHelper::CreateWithLease(master_view_key_, master_address,
                                          lease_id, version);

        if (ret == ErrorCode::OK) {
            // 成功成为 Leader
            return;
        }
        // 失败则重试
    }
}
```

**选举流程**：

```mermaid
sequenceDiagram
    participant N1 as 节点 1
    participant N2 as 节点 2
    participant E as etcd

    N1->>E: 检查 master_view key
    E-->>N1: key 不存在
    N2->>E: 检查 master_view key
    E-->>N2: key 不存在

    N1->>E: 获取 lease
    E-->>N1: lease_id = 123
    N2->>E: 获取 lease
    E-->>N2: lease_id = 456

    N1->>E: CreateWithLease(master_view, "N1", 123)
    E-->>N1: 成功！
    N2->>E: CreateWithLease(master_view, "N2", 456)
    E-->>N2: 失败 (key 已存在)

    Note over N1: 成为 Leader
    Note over N2: 等待 Leader 失效

    loop 保持 Leader
        N1->>E: KeepAlive(lease_id=123)
    end
```

### 3.7 本章小结

本章介绍了 Mooncake 的整体架构设计：

1. **分层架构**：控制平面（Conductor）、数据平面（P/D 节点）、存储层（Mooncake Store）、传输层（Transfer Engine）
2. **P/D 分离**：消除阶段干扰、支持独立资源配置和弹性伸缩
3. **Mooncake Store**：多级存储层次、树形 KVCache 组织、分片与复制
4. **Transfer Engine**：协议抽象、拓扑感知、多网卡聚合
5. **Conductor**：Cache-aware 调度、负载均衡、SLO 保证
6. **元数据管理**：基于 etcd、Leader 选举、高可用

---

[上一篇：核心概念与关键技术](../01-overview/02-core-concepts.md) | [目录](../README.md) | [下一篇：请求处理流程详解](02-request-lifecycle.md)
