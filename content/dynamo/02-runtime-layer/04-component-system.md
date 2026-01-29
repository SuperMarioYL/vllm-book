---
title: "Component 系统"
weight: 4
---

# Component 系统详解

> 本文详细介绍 Dynamo 的 Component 系统，包括 Namespace、Component、Endpoint 和 Registry 的设计与实现。

---

## 1. Component 系统概览

Component 系统是 Dynamo 分布式应用的核心抽象，定义了服务的组织方式：

```mermaid
graph TB
    subgraph system["Component 系统层级"]
        NS["Namespace<br/>命名空间"]
        COMP["Component<br/>组件"]
        EP["Endpoint<br/>端点"]
    end

    NS --> |包含多个| COMP
    COMP --> |包含多个| EP
```

---

## 2. Namespace 命名空间

### 2.1 设计目的

Namespace 提供了逻辑隔离，类似于 Kubernetes 的 Namespace：

```mermaid
graph TB
    subgraph prod["Namespace: production"]
        P_Frontend["Frontend"]
        P_Router["Router"]
        P_Worker["Worker"]
        P_Frontend --> P_Router --> P_Worker
    end

    subgraph dev["Namespace: development"]
        D_Frontend["Frontend"]
        D_Worker["Worker"]
        D_Frontend --> D_Worker
    end
```

### 2.2 使用方式

```python
# Python 中创建命名空间
namespace = runtime.namespace("production")
namespace_dev = runtime.namespace("development")
```

### 2.3 命名规范

Namespace 名称有严格的字符限制：

| 允许的字符 | 示例 |
|------------|------|
| 小写字母 | `a-z` |
| 数字 | `0-9` |
| 连字符 | `-` |
| 下划线 | `_` |

---

## 3. Component 组件模型

### 3.1 核心结构

```mermaid
classDiagram
    class Namespace {
        -runtime: DistributedRuntime
        -name: String
        +component(name) Component
    }

    class Component {
        -drt: DistributedRuntime
        -name: String
        -namespace: Namespace
        +endpoint(name) Endpoint
        +etcd_path() String
        +service_name() String
    }

    class Endpoint {
        -component: Component
        -name: String
        +client~Req,Resp~() Client
        +endpoint_builder() EndpointConfigBuilder
    }

    Namespace "1" --> "*" Component
    Component "1" --> "*" Endpoint
```

### 3.2 Python 中定义组件

```python
@service(dynamo={"namespace": "production"})
class VllmWorker:
    @dynamo_endpoint()
    async def generate(self, request):
        ...
```

### 3.3 路径与命名

Component 使用分层的命名体系：

| 属性 | 示例值 | 说明 |
|------|--------|------|
| namespace | `dynamo` | 命名空间 |
| name | `processor` | 组件名 |
| `etcd_path()` | `dynamo/components/processor` | etcd 存储路径 |
| `service_name()` | `dynamo\|processor` | NATS 服务名 |
| `path()` | `dynamo/processor` | 逻辑路径 |

### 3.4 事件发布/订阅

Component 支持事件驱动的通信：

```mermaid
sequenceDiagram
    participant CompA as Component A
    participant NATS as NATS
    participant CompB as Component B

    CompB->>NATS: subscribe("namespace.dynamo.component.a.kv_events")

    CompA->>NATS: publish("kv_events", event_data)
    NATS->>CompB: 消息推送

    CompB->>CompB: 处理事件
```

---

## 4. Endpoint 端点

### 4.1 Endpoint 定义

Endpoint 是 Component 上的可调用入口点：

```rust
pub struct Endpoint {
    component: Component,
    name: String,
}
```

### 4.2 端点引用格式

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

### 4.3 Endpoint 注册信息

端点注册到 etcd 时包含以下信息：

```json
{
    "component": "processor",
    "endpoint": "generate",
    "namespace": "dynamo",
    "lease_id": 123456789,
    "transport": { "nats_tcp": "dynamo|processor.generate-1a2b3c4d" }
}
```

### 4.4 Endpoint 启动流程

```mermaid
sequenceDiagram
    participant App as 应用
    participant EB as EndpointBuilder
    participant Reg as Registry
    participant NATS as NATS
    participant etcd as etcd
    participant Handler as Handler

    App->>EB: start()
    EB->>Reg: 获取 Service Group
    EB->>NATS: 创建 Service Endpoint
    EB->>Handler: 创建 PushEndpoint

    par 并行执行
        EB->>etcd: kv_create(endpoint_info)
    and
        Handler->>NATS: 开始监听请求
    end

    Note over Handler,NATS: 端点就绪，开始处理请求
```

---

## 5. Registry 注册中心

### 5.1 Registry 结构

```mermaid
graph TB
    subgraph registry["Component Registry"]
        REG["Registry"]

        subgraph services["Services Map"]
            SVC1["Service: dynamo|frontend"]
            SVC2["Service: dynamo|processor"]
            SVC3["Service: dynamo|worker"]
        end

        subgraph handlers["Stats Handlers Map"]
            SH1["Handler Map for frontend"]
            SH2["Handler Map for processor"]
            SH3["Handler Map for worker"]
        end
    end

    REG --> SVC1
    REG --> SVC2
    REG --> SVC3

    REG --> SH1
    REG --> SH2
    REG --> SH3
```

### 5.2 Registry 解决的问题

| 问题 | 解决方案 |
|------|----------|
| **Service 重复创建** | 同一个 Component 的多个 Endpoint 共享一个 NATS Service |
| **Watch 重复** | 多个客户端连接同一个 Component 时，共享一个 etcd Watcher |
| **Stats 聚合** | 统一管理每个端点的统计信息处理器 |

### 5.3 没有 Registry vs 有 Registry

```mermaid
graph TB
    subgraph without["没有 Registry"]
        C1A["Client A to Component X"]
        C1B["Client B to Component X"]
        W1A["Watcher A for Component X"]
        W1B["Watcher B for Component X"]
    end

    subgraph with["有 Registry"]
        C2A["Client A to Component X"]
        C2B["Client B to Component X"]
        W2["Shared Watcher for Component X"]

        C2A --> W2
        C2B --> W2
    end
```

---

## 6. 组件生命周期

### 6.1 状态转换

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

### 6.2 完整生命周期

```mermaid
sequenceDiagram
    participant App as 应用
    participant Component as Component
    participant etcd as etcd
    participant NATS as NATS

    App->>Component: 创建组件
    Component->>etcd: 创建 Lease
    etcd-->>Component: Lease ID

    Component->>etcd: 注册端点
    Component->>NATS: 创建 Service

    loop 运行中
        Component->>etcd: Keep-Alive
        Component->>NATS: 处理请求
    end

    Note over App: 收到停止信号

    App->>Component: 停止
    Component->>NATS: 停止接收请求
    Component->>etcd: Revoke Lease

    Note over etcd: 自动删除注册信息
```

---

## 7. 使用示例

### 7.1 创建命名空间和组件

```python
from dynamo.runtime import DistributedRuntime

async def main():
    runtime = await DistributedRuntime.create()

    # 创建命名空间
    namespace = runtime.namespace("production")

    # 创建组件
    component = namespace.component("worker")

    # 创建端点
    endpoint = component.endpoint("generate")
```

### 7.2 定义服务

```python
@service(dynamo={"namespace": "production"})
class VllmWorker:
    @dynamo_endpoint()
    async def generate(self, request):
        # 处理请求
        async for token in self.engine.generate(request):
            yield token
```

### 7.3 服务配置

```yaml
VllmWorker:
  model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  enable-prefix-caching: true
  ServiceArgs:
    workers: 4
    resources:
      gpu: 1
```

---

## 小结

本文介绍了 Dynamo 的 Component 系统：

1. **Namespace**：逻辑隔离，多租户支持
2. **Component**：可部署的服务单元
3. **Endpoint**：可调用的服务接口
4. **Registry**：资源复用和统一管理

---

## 下一篇

继续阅读 [05-Pipeline 架构](05-pipeline-architecture.md)，了解 Dynamo 的数据流处理机制。
