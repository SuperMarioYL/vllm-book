---
title: "系统架构总览"
weight: 4
---

# 系统架构总览

> 本文介绍 Dynamo 的五层架构模型、核心概念定义以及数据流与控制流。

---

## 1. 五层架构模型

Dynamo 采用经典的五层架构：

```mermaid
graph TB
    subgraph app["应用层"]
        HTTP["HTTP Server<br/>OpenAI Compatible API"]
    end

    subgraph sched["调度层"]
        FE["Frontend"]
        PROC["Processor"]
        ROUTER["KV-Aware Router"]
    end

    subgraph infer["推理层"]
        W1["Worker 1<br/>vLLM/TRT-LLM/SGLang"]
        W2["Worker 2"]
        W3["Worker N"]
        PW["Prefill Worker"]
    end

    subgraph runtime["运行时层"]
        RT["Distributed Runtime"]
        REG["Component Registry"]
        PIPE["Pipeline Engine"]
    end

    subgraph comm["通信层"]
        ETCD[("etcd<br/>服务发现")]
        NATS[("NATS<br/>事件总线")]
        NIXL["NIXL<br/>KV 传输"]
    end

    HTTP --> FE
    FE --> PROC
    PROC --> ROUTER
    ROUTER --> W1
    ROUTER --> W2
    ROUTER --> W3
    W1 --> PW

    W1 --> RT
    W2 --> RT
    W3 --> RT
    PW --> RT

    RT --> ETCD
    RT --> NATS
    W1 -.-> NIXL
    PW -.-> NIXL
```

---

## 2. 各层职责

| 层 | 组件 | 职责 |
|-----|------|------|
| **应用层** | HTTP Server | 提供 OpenAI 兼容 API |
| **调度层** | Frontend, Processor, Router | 请求处理、tokenization、智能路由 |
| **推理层** | Workers | 实际的模型推理 |
| **运行时层** | Runtime, Registry | 组件生命周期、服务发现 |
| **通信层** | etcd, NATS, NIXL | 协调、事件、数据传输 |

---

## 3. 核心概念定义

### 3.1 Namespace（命名空间）

逻辑隔离的组件集合，用于多租户或多环境：

```python
# 创建命名空间
namespace = runtime.namespace("production")
namespace_dev = runtime.namespace("development")
```

```mermaid
graph TB
    subgraph ns["命名空间隔离"]
        subgraph prod["production"]
            P1["Worker 1"]
            P2["Worker 2"]
        end

        subgraph dev["development"]
            D1["Worker 1"]
            D2["Worker 2"]
        end
    end
```

### 3.2 Component（组件）

可部署的服务单元，包含多个端点：

```python
# 定义组件
@service(dynamo={"namespace": "production"})
class VllmWorker:
    @dynamo_endpoint()
    async def generate(self, request):
        ...
```

### 3.3 Endpoint（端点）

组件暴露的可调用接口：

```yaml
# 端点引用格式
endpoint: dynamo.Processor.chat/completions
#         ^命名空间  ^组件     ^端点名
```

```mermaid
graph LR
    subgraph endpoint["端点结构"]
        NS["Namespace<br/>dynamo"] --> COMP["Component<br/>Processor"]
        COMP --> EP["Endpoint<br/>chat/completions"]
    end
```

### 3.4 Service（服务）

组件的运行实例，可以有多个副本：

```yaml
ServiceArgs:
  workers: 4  # 4 个服务实例
  resources:
    gpu: 1    # 每个实例 1 个 GPU
```

### 3.5 Pipeline（管道）

数据流处理管道，支持流式处理：

```mermaid
graph LR
    IN["输入"] --> OP1["Operator 1"]
    OP1 --> OP2["Operator 2"]
    OP2 --> OUT["输出流"]
```

---

## 4. 数据流与控制流

### 4.1 请求处理全链路

```mermaid
sequenceDiagram
    participant Client
    participant HTTP
    participant Frontend
    participant Processor
    participant Router
    participant Worker
    participant KVCache

    Client->>HTTP: POST /v1/chat/completions
    HTTP->>Frontend: forward request
    Frontend->>Processor: process request

    Note over Processor: Tokenization
    Note over Processor: Chat Template

    Processor->>Router: route(token_ids)

    Note over Router: 查询 KV Index
    Note over Router: 计算匹配分数
    Note over Router: 选择最优 Worker

    Router->>Worker: generate(request)

    loop Token Generation
        Worker->>KVCache: read/write KV
        Worker-->>Frontend: stream token
        Frontend-->>HTTP: SSE event
        HTTP-->>Client: token
    end

    Worker-->>Frontend: finish
    Frontend-->>HTTP: [DONE]
    HTTP-->>Client: stream end
```

### 4.2 事件传播机制

系统使用 NATS 进行事件传播：

```mermaid
graph TB
    subgraph produce["事件生产"]
        W1["Worker 1"] --> |kv_stored| NATS[("NATS")]
        W2["Worker 2"] --> |kv_stored| NATS
        W1 --> |load_metrics| NATS
        W2 --> |load_metrics| NATS
    end

    subgraph consume["事件消费"]
        NATS --> |kv_events| IDX["KV Indexer"]
        NATS --> |load_metrics| AGG["Metrics Aggregator"]
        IDX --> ROUTER["Router"]
        AGG --> ROUTER
    end
```

### 4.3 状态同步策略

Dynamo 使用**最终一致性**模型：

```mermaid
graph LR
    subgraph sync["状态同步"]
        LOCAL["本地状态"] --> |定期发布| NATS[("NATS")]
        NATS --> |广播| OTHER["其他节点"]
        OTHER --> |更新| REMOTE["远程缓存"]
    end
```

**一致性模型选择**：
- **强一致需求**（服务注册）：使用 etcd
- **最终一致需求**（KV 索引）：使用 NATS 事件

---

## 5. 组件交互示意

### 5.1 控制平面 vs 数据平面

```mermaid
graph TB
    subgraph control["控制平面"]
        ETCD[("etcd")]
        REG["服务注册"]
        DISC["服务发现"]

        REG --> ETCD
        ETCD --> DISC
    end

    subgraph event["事件平面"]
        NATS[("NATS")]
        PUB["发布者"]
        SUB["订阅者"]

        PUB --> NATS
        NATS --> SUB
    end

    subgraph data["数据平面"]
        NIXL["NIXL"]
        GPU1["GPU 1"]
        GPU2["GPU 2"]

        GPU1 --> |RDMA| GPU2
    end
```

### 5.2 组件生命周期

```mermaid
stateDiagram-v2
    [*] --> Created: 创建
    Created --> Registered: 注册到 etcd
    Registered --> Running: 启动服务
    Running --> Healthy: 健康检查通过
    Healthy --> Running: 持续运行
    Running --> Stopping: 收到停止信号
    Stopping --> Deregistered: 从 etcd 注销
    Deregistered --> [*]
```

---

## 6. 架构优势

| 特性 | 说明 | 收益 |
|------|------|------|
| **分层架构** | 各层职责清晰 | 易于维护和扩展 |
| **服务发现** | 基于 etcd | 动态扩缩容 |
| **事件驱动** | 基于 NATS | 松耦合、高可扩展 |
| **零拷贝传输** | 基于 NIXL | 最小化延迟 |
| **多引擎支持** | 抽象接口 | 灵活选择后端 |

---

## 下一篇

继续阅读 [05-部署模式详解](05-deployment-modes.md)，了解 Dynamo 的四种部署模式及其适用场景。
