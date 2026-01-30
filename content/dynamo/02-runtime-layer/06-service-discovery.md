---
title: "服务发现机制"
weight: 6
---

> 本文详细介绍 Dynamo 的服务发现机制，包括 etcd 租约管理、端点注册和 Watch 监听。

---

## 1. 服务发现概述

Dynamo 使用 **客户端发现模式**，通过 etcd 实现服务注册和发现：

```mermaid
graph TB
    subgraph discovery["服务发现架构"]
        etcd[("etcd<br/>注册中心")]

        W1["Worker 1"]
        W2["Worker 2"]
        Router["Router"]
        Frontend["Frontend"]
    end

    W1 --> |注册| etcd
    W2 --> |注册| etcd

    Router --> |Watch| etcd
    Frontend --> |查询| etcd

    Router --> W1
    Router --> W2
```

---

## 2. etcd 租约管理

### 2.1 Lease 创建

每个服务实例创建一个 Lease，用于健康检查：

```mermaid
sequenceDiagram
    participant Client as Dynamo Client
    participant etcd as etcd Server

    Client->>etcd: LeaseGrant(TTL=10s)
    etcd-->>Client: Lease ID

    Client->>etcd: Put(key, value, lease_id)
    etcd-->>Client: OK

    Note over Client,etcd: Key 绑定到 Lease
```

### 2.2 Keep-Alive 机制

Lease 创建后，需要定期发送 Keep-Alive 来保持活跃：

```mermaid
sequenceDiagram
    participant Client as Dynamo Client
    participant etcd as etcd Server

    Client->>etcd: LeaseGrant(TTL=10s)
    etcd-->>Client: Lease ID

    loop 每 TTL/3 秒
        Client->>etcd: LeaseKeepAlive(ID)
        etcd-->>Client: TTL Refreshed
    end

    Note over Client: CancellationToken 触发

    Client->>etcd: LeaseRevoke(ID)
    Note over etcd: 删除所有绑定的 Key
```

### 2.3 Lease 结构

```rust
pub struct Lease {
    id: i64,
    cancel_token: CancellationToken,
}

impl Lease {
    /// 获取 Lease ID
    pub fn id(&self) -> i64;

    /// 获取主取消令牌（取消会撤销 Lease）
    pub fn primary_token(&self) -> CancellationToken;

    /// 获取子取消令牌（取消不会撤销 Lease）
    pub fn child_token(&self) -> CancellationToken;

    /// 撤销 Lease
    pub fn revoke(&self);
}
```

---

## 3. 端点注册

### 3.1 注册信息结构

```rust
pub struct ComponentEndpointInfo {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub lease_id: i64,
    pub transport: TransportType,
}

pub enum TransportType {
    NatsTcp(String),  // NATS 主题
}
```

### 3.2 etcd 中的存储

```
Key: dynamo/components/processor/generate:1a2b3c4d
Value: {
    "component": "processor",
    "endpoint": "generate",
    "namespace": "dynamo",
    "lease_id": 123456789,
    "transport": { "nats_tcp": "dynamo|processor.generate-1a2b3c4d" }
}
```

### 3.3 原子创建

使用 etcd 事务确保 Key 不存在时才创建：

```mermaid
graph LR
    subgraph txn["etcd 事务"]
        WHEN["when: version == 0<br/>Key 不存在"]
        THEN["then: PUT key value"]
        ELSE["else: 返回错误"]

        WHEN --> |满足| THEN
        WHEN --> |不满足| ELSE
    end
```

---

## 4. Watch 监听

### 4.1 Watch 工作流程

```mermaid
sequenceDiagram
    participant Client as Dynamo Client
    participant etcd as etcd Server
    participant Cache as 本地缓存

    Client->>etcd: GET /dynamo/components/* (with prefix)
    etcd-->>Client: [Endpoint A, Endpoint B]
    Client->>Cache: 初始化缓存

    Client->>etcd: WATCH /dynamo/components/* (from revision N)

    Note over etcd: 新 Endpoint 注册
    etcd->>Client: PUT event (Endpoint C)
    Client->>Cache: 添加 Endpoint C

    Note over etcd: Endpoint 下线
    etcd->>Client: DELETE event (Endpoint A)
    Client->>Cache: 移除 Endpoint A
```

### 4.2 PrefixWatcher

```rust
pub struct PrefixWatcher {
    prefix: String,
    watcher: Watcher,
    rx: Receiver<WatchEvent>,
}

pub enum WatchEvent {
    Put(KeyValue),
    Delete(KeyValue),
}
```

### 4.3 Watch 关键设计

| 设计点 | 说明 |
|--------|------|
| **revision 连续性** | 从 GET 返回的 revision + 1 开始 Watch，确保不遗漏事件 |
| **prev_key** | 删除事件时包含被删除的 Key 信息 |
| **Secondary Runtime** | Watch 循环在 Secondary Runtime 中运行，不影响主业务 |

---

## 5. 服务发现完整流程

### 5.1 服务注册流程

```mermaid
sequenceDiagram
    participant Service as 服务实例
    participant DRT as DistributedRuntime
    participant etcd as etcd
    participant NATS as NATS

    Service->>DRT: 创建 Component
    DRT->>etcd: 创建 Lease
    etcd-->>DRT: Lease ID

    Service->>DRT: 创建 Endpoint
    DRT->>NATS: 创建 Service Endpoint
    DRT->>etcd: 注册 Endpoint 信息

    loop Keep-Alive
        DRT->>etcd: LeaseKeepAlive
    end
```

### 5.2 服务发现流程

```mermaid
sequenceDiagram
    participant Router as Router
    participant DRT as DistributedRuntime
    participant etcd as etcd
    participant Cache as 端点缓存

    Router->>DRT: 创建 Endpoint Client
    DRT->>etcd: GET + WATCH endpoints

    etcd-->>DRT: 当前端点列表
    DRT->>Cache: 初始化缓存

    loop Watch 事件
        etcd->>DRT: PUT/DELETE 事件
        DRT->>Cache: 更新缓存
    end

    Router->>Cache: 获取可用端点
    Cache-->>Router: [Endpoint 1, Endpoint 2]
```

### 5.3 服务下线流程

```mermaid
sequenceDiagram
    participant Service as 服务实例
    participant DRT as DistributedRuntime
    participant etcd as etcd
    participant Router as Router

    Note over Service: 收到停止信号

    Service->>DRT: 停止服务
    DRT->>etcd: LeaseRevoke

    Note over etcd: Lease 过期<br/>删除绑定的 Key

    etcd->>Router: DELETE 事件
    Router->>Router: 从路由表移除
```

---

## 6. 最佳实践

### 6.1 Lease TTL 设置

| 场景 | 推荐 TTL | 原因 |
|------|----------|------|
| 稳定网络 | 10s | 快速检测故障 |
| 不稳定网络 | 30s | 避免频繁超时 |
| 开发测试 | 5s | 快速迭代 |

### 6.2 Watch 重连处理

```mermaid
graph TB
    subgraph reconnect["Watch 重连策略"]
        WATCH["Watch 连接"]
        DISC["连接断开"]
        RETRY["重试"]
        FULL["全量同步"]

        WATCH --> DISC
        DISC --> RETRY
        RETRY --> |"revision 有效"| WATCH
        RETRY --> |"revision 过期"| FULL
        FULL --> WATCH
    end
```

### 6.3 健康检查集成

```mermaid
graph LR
    subgraph health["健康检查"]
        Lease["Lease Keep-Alive"]
        App["应用健康检查"]
        Combined["综合健康状态"]
    end

    Lease --> Combined
    App --> Combined
    Combined --> |不健康| Revoke["撤销 Lease"]
```

---

## 小结

本文介绍了 Dynamo 的服务发现机制：

1. **etcd 租约**：Lease 创建和 Keep-Alive
2. **端点注册**：原子创建，绑定 Lease
3. **Watch 监听**：实时感知服务变化
4. **完整流程**：注册、发现、下线

---

## 下一章

完成本章阅读后，建议继续阅读 [第三章：LLM 推理层原理](../03-llm-inference-layer/README.md)，了解 Dynamo 如何支持多种推理引擎。
