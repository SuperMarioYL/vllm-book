---
title: "vLLM 架构概览"
weight: 3
---


> **阅读时间**: 约 20 分钟
> **前置要求**: [KV Cache 核心概念](./02-kv-cache-fundamentals.md)

---

## 概述

vLLM 是目前最流行的 LLM 推理框架之一。UCM 通过与 vLLM 集成来实现 KV Cache 管理优化。本文介绍 vLLM 的核心架构和关键概念。

---

## 1. vLLM 简介

### 1.1 设计目标

vLLM 的核心目标是解决 LLM 推理中的内存效率问题：

```mermaid
graph TB
    subgraph goals["vLLM 设计目标"]
        G1["高吞吐<br/>最大化 GPU 利用率"]
        G2["低延迟<br/>快速响应用户请求"]
        G3["内存高效<br/>支持更多并发请求"]
    end

    subgraph solutions["核心技术"]
        S1["PagedAttention<br/>分页式 KV Cache 管理"]
        S2["Continuous Batching<br/>连续批处理"]
        S3["Optimized Kernels<br/>优化的 CUDA 内核"]
    end

    goals --> solutions
```
### 1.2 与其他框架对比
| 特性 | vLLM | HuggingFace | TensorRT-LLM |
|------|------|-------------|--------------|
| PagedAttention | Yes | No | Partial |
| 连续批处理 | Yes | No | Yes |
| 内存效率 | 高 | 低 | 中 |
| 易用性 | 高 | 最高 | 中 |
| UCM 集成 | 支持 | - | - |
---
## 2. vLLM 整体架构

### 2.1 架构层次

```mermaid
graph TB
    subgraph api["API 层"]
        A1["OpenAI 兼容 API"]
        A2["Python SDK"]
    end

    subgraph engine["引擎层"]
        E1["LLMEngine<br/>请求管理和调度"]
    end

    subgraph scheduler["调度层"]
        S1["Scheduler<br/>请求调度策略"]
        S2["BlockManager<br/>Block 分配管理"]
    end

    subgraph worker["执行层"]
        W1["Worker<br/>模型执行"]
        W2["ModelRunner<br/>前向计算"]
    end

    subgraph model["模型层"]
        M1["Model<br/>Transformer 模型"]
        M2["Attention<br/>注意力计算"]
    end

    api --> engine
    engine --> scheduler
    scheduler --> worker
    worker --> model
```

### 2.2 核心组件

| 组件 | 职责 | 关键文件 |
|------|------|----------|
| LLMEngine | 请求生命周期管理 | `vllm/engine/llm_engine.py` |
| Scheduler | 请求调度和批处理 | `vllm/core/scheduler.py` |
| BlockManager | KV Cache Block 分配 | `vllm/core/block_manager.py` |
| Worker | GPU 上的模型执行 | `vllm/worker/worker.py` |
| ModelRunner | 前向计算封装 | `vllm/worker/model_runner.py` |

---
## 3. PagedAttention 原理
### 3.1 传统 KV Cache 的问题
传统方法为每个请求预分配固定大小的连续内存：
```mermaid
graph TB
    subgraph traditional["传统 KV Cache 管理"]
        subgraph req1["请求 1"]
            R1A["已使用"] --> R1B["空闲（预留）"]
        end
        subgraph req2["请求 2"]
            R2A["已使用"] --> R2B["空闲（预留）"]
        end
        subgraph req3["请求 3"]
            R3A["已使用"] --> R3B["空闲（预留）"]
        end

        Problem["问题: 内存碎片和浪费"]
    end
```

**问题**:
- 必须预分配最大长度
- 无法准确预知每个请求的实际长度
- 导致大量内存浪费
### 3.2 PagedAttention 的解决方案
PagedAttention 借鉴操作系统的虚拟内存思想，将 KV Cache 分成固定大小的 Block：
```mermaid
graph TB
    subgraph paged["PagedAttention"]
        subgraph logical["逻辑视图 - 连续"]
            L1["Token 0-15"] --> L2["Token 16-31"] --> L3["Token 32-47"]
        end

        subgraph physical["物理视图 - 分散"]
            P1["Block 3"]
            P2["Block 7"]
            P3["Block 1"]
        end

        subgraph mapping["Block Table 映射"]
            M["逻辑块 0 → 物理块 3<br/>逻辑块 1 → 物理块 7<br/>逻辑块 2 → 物理块 1"]
        end

        logical --> mapping
        mapping --> physical
    end
```
### 3.3 Block 的结构
```python
class Block:
    block_size = 16  # 每个 Block 存储 16 个 token 的 KV
    # 物理内存布局
    # K: [block_size, num_heads, head_dim]
    # V: [block_size, num_heads, head_dim]
```

**优势**:
- 按需分配，无需预留
- 减少内存碎片
- 支持更多并发请求
---
## 4. Scheduler 调度器

### 4.1 调度器职责

```mermaid
graph TB
    subgraph scheduler["Scheduler 职责"]
        S1["接收新请求<br/>加入等待队列"]
        S2["选择执行请求<br/>基于优先级和资源"]
        S3["分配 KV Block<br/>通过 BlockManager"]
        S4["构建批次<br/>打包多个请求"]
        S5["生成调度输出<br/>SchedulerOutput"]
    end

    S1 --> S2 --> S3 --> S4 --> S5
```

### 4.2 请求状态

```mermaid
stateDiagram-v2
    [*] --> WAITING: 新请求到达
    WAITING --> RUNNING: 被调度执行
    RUNNING --> RUNNING: 生成 token
    RUNNING --> WAITING: 被抢占
    RUNNING --> FINISHED: 生成完成
    RUNNING --> FINISHED: 达到长度限制
    FINISHED --> [*]
```

### 4.3 调度策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| FCFS | 先来先服务 | 默认策略 |
| 抢占式 | 长请求可被短请求抢占 | 低延迟优先 |
| 公平调度 | 均衡各请求的进度 | 多用户场景 |

---
## 5. Worker 和 ModelRunner
### 5.1 Worker 职责
Worker 负责在 GPU 上执行模型推理：
```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant Worker as Worker
    participant Runner as ModelRunner
    participant Model as Model

    Sched->>Worker: SchedulerOutput
    Worker->>Worker: 准备输入数据
    Worker->>Runner: execute_model()
    Runner->>Model: forward()
    Model-->>Runner: logits
    Runner->>Runner: sample()
    Runner-->>Worker: SamplerOutput
    Worker-->>Sched: 返回结果
```

### 5.2 ModelRunner 执行流程

```python
# ModelRunner 简化流程
class ModelRunner:
    def execute_model(self, scheduler_output):
        # 1. 准备输入张量
        input_ids = self._prepare_inputs(scheduler_output)
        # 2. 准备 KV Cache
        kv_caches = self._get_kv_caches()

        # 3. 执行前向计算
        logits = self.model.forward(
            input_ids=input_ids,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata
        )
        # 4. 采样生成 token
        output = self.sampler(logits)

        return output
```
---
## 6. KVConnector 接口

### 6.1 接口设计

vLLM 提供 KVConnector 接口，允许外部系统（如 UCM）接入 KV Cache 管理：

```mermaid
classDiagram
    class KVConnectorBase {
        <<abstract>>
        +get_num_new_matched_tokens()
        +build_connector_meta()
        +bind_connector_metadata()
        +start_load_kv()
        +wait_for_layer_load()
        +save_kv_layer()
        +wait_for_save()
    }

    class UCMDirectConnector {
        +store: UcmKVStoreBase
        +request_hasher: RequestHasher
        +get_num_new_matched_tokens()
        +start_load_kv()
        ...
    }

    KVConnectorBase <|-- UCMDirectConnector
```

### 6.2 接口方法说明

| 方法 | 调用位置 | 说明 |
|------|----------|------|
| `get_num_new_matched_tokens` | Scheduler | 查询 KV Cache 命中数量 |
| `build_connector_meta` | Scheduler | 构建传输元数据 |
| `bind_connector_metadata` | Worker | 绑定元数据到执行上下文 |
| `start_load_kv` | Worker | 开始加载 KV Cache |
| `wait_for_layer_load` | Worker | 等待层级加载完成 |
| `save_kv_layer` | Worker | 保存 KV Cache |
| `wait_for_save` | Worker | 等待保存完成 |

### 6.3 UCM 如何接入

```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant UCM as UCMConnector
    participant Store as UcmKVStore

    Note over Sched,Store: 查询阶段
    Sched->>UCM: get_num_new_matched_tokens()
    UCM->>UCM: 生成 Block Hash
    UCM->>Store: lookup(block_ids)
    Store-->>UCM: 命中结果
    UCM-->>Sched: 返回命中数量

    Note over Sched,Store: 加载阶段
    Sched->>UCM: build_connector_meta()
    UCM-->>Sched: 传输元数据

    Note over Sched,Store: Worker 执行
    UCM->>Store: load(block_ids)
    Store-->>UCM: KV 数据
```

---
## 7. Block 管理
### 7.1 BlockManager
BlockManager 管理物理 Block 的分配和释放：
```mermaid
graph TB
    subgraph blockmanager["BlockManager"]
        subgraph pools["Block 池"]
            GPU["GPU Block Pool<br/>物理 KV 存储"]
            CPU["CPU Block Pool<br/>Swap 使用"]
        end
        subgraph ops["操作"]
            Alloc["allocate()<br/>分配 Block"]
            Free["free()<br/>释放 Block"]
            Swap["swap()<br/>GPU↔CPU 交换"]
        end
    end
```
### 7.2 Block Table
每个请求维护一个 Block Table，记录逻辑块到物理块的映射：
```python
request_1 = {
    "block_table": [5, 12, 3, None, None],  # 逻辑块 → 物理块
    "num_tokens": 45,  # 当前 token 数
    "max_blocks": 5    # 最大 Block 数
}

# 逻辑块 1 → 物理块 12 (Token 16-31)
# 逻辑块 2 → 物理块 3 (Token 32-44)
```

### 7.3 Copy-on-Write

当多个请求共享前缀时，vLLM 使用 Copy-on-Write 优化：

```mermaid
graph TB
    subgraph cow["Copy-on-Write"]
        subgraph before["修改前 - 共享"]
            R1["请求 1"] --> B1["Block 5"]
            R2["请求 2"] --> B1
        end

        subgraph after["修改后 - 复制"]
            R1A["请求 1"] --> B1A["Block 5<br/>(原内容)"]
            R2A["请求 2"] --> B2A["Block 8<br/>(新内容)"]
        end
        before --> |"请求 2 需要写入"| after
    end
```
---
## 8. vLLM 扩展点

### 8.1 UCM 使用的扩展点

```mermaid
graph TB
    subgraph extension["vLLM 扩展点"]
        E1["KVConnector<br/>KV Cache 管理接口"]
        E2["Attention Layer<br/>注意力计算钩子"]
        E3["Scheduler Output<br/>调度输出扩展"]
        E4["Model Runner<br/>执行流程钩子"]
    end

    subgraph ucm["UCM 利用方式"]
        U1["UCMDirectConnector<br/>实现 KV 加载/保存"]
        U2["Sparse Attention<br/>稀疏注意力替换"]
        U3["Metadata 传递<br/>Block Hash 等信息"]
        U4["Monkey Patch<br/>注入自定义逻辑"]
    end

    E1 --> U1
    E2 --> U2
    E3 --> U3
    E4 --> U4
```

### 8.2 Monkey Patching

UCM 通过 Monkey Patching 在不修改 vLLM 源码的情况下注入功能：

```python
import vllm.attention

_original_attention_forward = vllm.attention.Attention.forward

def _patched_attention_forward(self, ...):
    # UCM 前置处理
    ucm_sparse.attention_begin(...)

    # 调用原始函数
    result = _original_attention_forward(self, ...)

    # UCM 后置处理
    ucm_sparse.attention_finished(...)

    return result

vllm.attention.Attention.forward = _patched_attention_forward
```

---
## 9. 关键概念总结
| 概念 | 说明 | 与 UCM 的关系 |
|------|------|--------------|
| PagedAttention | 分页式 KV Cache 管理 | UCM 基于 Block 级别操作 |
| Scheduler | 请求调度和资源分配 | UCM 在此查询 KV 命中 |
| Worker | GPU 上的模型执行 | UCM 在此加载/保存 KV |
| KVConnector | KV Cache 管理接口 | UCM 实现此接口 |
| Block Table | 逻辑块到物理块映射 | UCM 扩展为内容寻址 |
| Monkey Patching | 运行时代码注入 | UCM 的非侵入式集成方式 |

---

## 延伸阅读

- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm)
