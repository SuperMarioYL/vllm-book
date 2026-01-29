---
title: "三平面架构"
weight: 2
---

# 三平面架构设计

> Dynamo 采用三平面架构，将不同类型的通信需求分离到不同的基础设施组件中。本文档解析这一设计的原理和优势。

---

## 1. 架构总览

### 1.1 三平面定义

```mermaid
graph TB
    subgraph planes["Dynamo 三平面架构"]
        subgraph control["控制平面 - etcd"]
            E1["服务注册"]
            E2["配置管理"]
            E3["分布式协调"]
            E4["租约管理"]
        end

        subgraph event["事件平面 - NATS"]
            N1["KV 事件广播"]
            N2["负载指标收集"]
            N3["Prefill 工作队列"]
            N4["服务状态同步"]
        end

        subgraph data["数据平面 - NIXL/TCP"]
            D1["KV Cache 传输"]
            D2["响应流传输"]
            D3["大数据块移动"]
        end

        Worker1["Worker 1"] --> E1
        Worker1 --> N1
        Worker1 --> D1

        Worker2["Worker 2"] --> E1
        Worker2 --> N2
        Worker2 --> D1

        Router["Router"] --> E2
        Router --> N1
        Router --> N2
    end
```

### 1.2 职责对比

| 平面 | 技术选型 | 主要职责 | 数据特点 |
|------|----------|----------|----------|
| 控制平面 | etcd | 服务发现、配置管理、租约 | 小数据量、强一致性 |
| 事件平面 | NATS | 事件广播、指标收集、工作队列 | 中等数据量、高吞吐 |
| 数据平面 | NIXL/TCP | KV Cache 传输、流式响应 | 大数据量、低延迟 |

---

## 2. 控制平面

### 2.1 职责范围

控制平面负责系统的"元数据"管理：

```mermaid
graph LR
    subgraph control["控制平面职责"]
        A["服务注册"] --> E["etcd"]
        B["服务发现"] --> E
        C["配置存储"] --> E
        D["租约/心跳"] --> E
    end
```

### 2.2 数据特点

- **数据量小**：KB 级别的元数据
- **更新频率低**：秒级到分钟级
- **一致性要求高**：必须准确

### 2.3 为什么选择 etcd

| 特性 | etcd | 其他选项 |
|------|------|----------|
| 一致性 | Raft 强一致 | ZooKeeper (ZAB) |
| Watch | 原生支持 | ✓ |
| 租约 | 内置支持 | 需要自实现 |
| 运维 | 简单 | 复杂 |

---

## 3. 事件平面

### 3.1 职责范围

事件平面负责系统内的"事件流"：

```mermaid
graph LR
    subgraph event["事件平面职责"]
        A["KV 事件"] --> N["NATS"]
        B["负载指标"] --> N
        C["工作队列"] --> N
        D["状态通知"] --> N
    end
```

### 3.2 数据特点

- **数据量中等**：消息级别
- **更新频率高**：毫秒到秒级
- **可靠性要求**：允许偶尔丢失（Core NATS）或必须可靠（JetStream）

### 3.3 为什么选择 NATS

| 特性 | NATS | Kafka | RabbitMQ |
|------|------|-------|----------|
| 延迟 | 微秒级 | 毫秒级 | 毫秒级 |
| 部署 | 单二进制 | 依赖 ZK | 依赖 Erlang |
| 协议 | 简单文本 | 复杂二进制 | AMQP |
| 持久化 | JetStream | 内置 | 内置 |

---

## 4. 数据平面

### 4.1 职责范围

数据平面负责大规模数据传输：

```mermaid
graph LR
    subgraph dataplane["数据平面职责"]
        A["KV Cache"] --> T["NIXL/TCP"]
        B["流式响应"] --> T
        C["模型权重"] --> T
    end
```

### 4.2 数据特点

- **数据量大**：MB 到 GB 级别
- **延迟敏感**：影响 TTFT
- **带宽密集**：需要高吞吐

### 4.3 技术选择

| 场景 | 技术 | 原因 |
|------|------|------|
| KV Cache 传输 | NIXL (RDMA) | 低延迟、高带宽 |
| 流式响应 | TCP Pipeline | 广泛兼容 |
| 无 RDMA 环境 | TCP 回退 | 通用性 |

---

## 5. 平面间协作

### 5.1 完整请求流程

```mermaid
sequenceDiagram
    participant R as Router
    participant E as etcd
    participant N as NATS
    participant PW as Prefill Worker
    participant DW as Decode Worker
    participant NIXL as NIXL

    Note over R,E: 控制平面 - 服务发现
    R->>E: 查询可用 Workers
    E-->>R: 返回 Worker 列表

    Note over R,N: 事件平面 - 调度决策
    N-->>R: 接收负载指标
    R->>R: 选择最佳 Worker

    Note over R,N: 事件平面 - 任务分发
    R->>N: 发布 Prefill 请求
    N-->>PW: 推送到 Prefill Queue

    Note over PW,NIXL: 数据平面 - KV 传输
    PW->>NIXL: 读取已有 KV（如果有）
    PW->>PW: 执行 Prefill 计算
    PW->>NIXL: 写入计算结果

    Note over PW,DW: 通知 Decode Worker
    PW->>DW: Prefill 完成通知
```

### 5.2 分离式 Prefill 流程

```mermaid
sequenceDiagram
    participant Client as Client
    participant Router as Router
    participant NATS as NATS
    participant DW as Decode Worker
    participant PW as Prefill Worker
    participant etcd as etcd
    participant NIXL as NIXL

    Note over Client,NIXL: 1. 请求到达
    Client->>Router: 请求（长 Prompt）

    Note over Router,DW: 2. 路由决策
    Router->>Router: 检查 prompt_len > threshold?
    Router->>Router: 决定使用分离式 Prefill

    Note over Router,DW: 3. 分配资源
    Router->>DW: 分配 KV 块
    DW-->>Router: 返回块信息

    Note over Router,NATS: 4. 任务入队
    Router->>NATS: enqueue RemotePrefillRequest

    Note over NATS,PW: 5. Prefill 处理
    PW->>NATS: dequeue 获取任务
    NATS-->>PW: RemotePrefillRequest

    PW->>etcd: 查询 Decode Worker NIXL 元数据
    etcd-->>PW: NixlMetadata

    PW->>PW: 执行 Prefill 计算

    Note over PW,DW: 6. KV Cache 传输
    PW->>NIXL: RDMA Write KV Cache
    NIXL->>DW: 直接写入 GPU 内存

    Note over PW,DW: 7. 完成通知
    PW->>DW: Prefill 完成通知

    Note over DW,Client: 8. Decode 阶段
    DW->>DW: 执行 Decode
    DW-->>Client: 流式响应
```

---

## 6. 设计优势

### 6.1 关注点分离

每个平面专注于特定类型的通信：

```
控制平面：强一致性，低频更新
     ↓
事件平面：高吞吐，最终一致
     ↓
数据平面：低延迟，大带宽
```

### 6.2 独立扩展

```mermaid
graph TB
    subgraph scale["独立扩展能力"]
        E["etcd 集群<br/>3-5 节点即可"]
        N["NATS 集群<br/>按消息量扩展"]
        D["NIXL<br/>按 GPU 数量扩展"]
    end
```

### 6.3 故障隔离

| 故障场景 | 影响范围 | 恢复策略 |
|----------|----------|----------|
| etcd 不可用 | 新服务无法注册 | 现有服务继续运行 |
| NATS 不可用 | 事件延迟 | 本地缓存兜底 |
| NIXL 故障 | KV 传输失败 | 回退到 TCP |

---

## 7. 配置建议

### 7.1 小规模部署（< 10 GPU）

```yaml
control_plane:
  etcd:
    nodes: 1  # 单节点即可
    memory: 2GB

event_plane:
  nats:
    nodes: 1
    memory: 1GB

data_plane:
  nixl:
    enabled: false  # 使用 TCP
```

### 7.2 中规模部署（10-50 GPU）

```yaml
control_plane:
  etcd:
    nodes: 3  # 3 节点集群
    memory: 4GB

event_plane:
  nats:
    nodes: 3
    memory: 2GB
    jetstream: true

data_plane:
  nixl:
    enabled: true
    fallback: tcp
```

### 7.3 大规模部署（50+ GPU）

```yaml
control_plane:
  etcd:
    nodes: 5
    memory: 8GB
    ssd: true

event_plane:
  nats:
    nodes: 5
    memory: 4GB
    jetstream: true

data_plane:
  nixl:
    enabled: true
    rdma_interface: ib0
```

---

## 总结

三平面架构的核心思想：

1. **关注点分离**：不同通信需求使用不同技术
2. **技术适配**：每个平面选择最合适的技术
3. **独立扩展**：各平面可以独立扩容
4. **故障隔离**：一个平面故障不影响其他平面

这套架构使 Dynamo 能够在保证可靠性的同时，实现高性能的分布式推理。

---

## 参考资料

- [etcd Documentation](https://etcd.io/docs/)
- [NATS Documentation](https://docs.nats.io/)
- [NVIDIA NIXL](https://developer.nvidia.com/nixl)
