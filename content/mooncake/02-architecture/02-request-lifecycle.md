---
title: "请求处理流程详解"
weight: 2
---

[上一篇：整体架构设计](01-system-architecture.md) | [目录](../README.md) | [下一篇：存储架构设计](../03-mooncake-store/01-storage-architecture.md)

---

# 请求处理流程详解

本章将深入分析一个请求从进入系统到返回结果的完整流程，涵盖各个组件之间的交互细节。

### 4.1 请求生命周期总览

一个 LLM 推理请求在 Mooncake 系统中经历以下主要阶段：

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Conductor as Conductor
    participant Prefill as Prefill Node
    participant Store as Mooncake Store
    participant Decode as Decode Node

    rect rgb(255, 240, 240)
        Note over Client,Conductor: 阶段 1 - 请求调度
        Client->>Conductor: 发送请求 (prompt)
        Conductor->>Conductor: Cache-aware 调度决策
        Conductor->>Prefill: 分配 Prefill 任务
    end

    rect rgb(240, 255, 240)
        Note over Prefill,Store: 阶段 2 - Prefill 处理
        Prefill->>Store: 查询 prefix cache
        Store-->>Prefill: 返回缓存的 KVCache (如有)
        Prefill->>Prefill: 计算 Prefill
        Prefill->>Store: 存储新生成的 KVCache
        Prefill->>Conductor: Prefill 完成
    end

    rect rgb(240, 240, 255)
        Note over Conductor,Decode: 阶段 3 - Decode 调度与执行
        Conductor->>Decode: 分配 Decode 任务
        Decode->>Store: 加载 KVCache
        loop 自回归生成
            Decode->>Decode: 生成一个 token
            Decode-->>Client: 流式返回 token
            Decode->>Store: 更新 KVCache (可选)
        end
    end

    rect rgb(255, 255, 240)
        Note over Decode,Client: 阶段 4 - 完成与清理
        Decode->>Conductor: Decode 完成
        Decode-->>Client: 结束标记
    end
```

### 4.2 阶段一：请求调度

#### 4.2.1 请求接收与解析

当客户端发送请求时，Conductor 首先解析请求内容：

```python
# 请求结构示例
request = {
    "prompt": "Please analyze this document: [doc content]...",
    "max_tokens": 2048,
    "temperature": 0.7,
    "stream": True
}
```

Conductor 将 prompt 进行 tokenize，并计算其 **prefix hash**：

```mermaid
graph LR
    P[Prompt 文本] --> T[Tokenizer]
    T --> Tokens[Token IDs]
    Tokens --> H[Hash 函数]
    H --> PH[Prefix Hash]
```

#### 4.2.2 Cache 查询

Conductor 查询 Mooncake Store，获取各节点上已缓存的 prefix 情况：

```mermaid
graph TB
    subgraph "Mooncake Store 元数据"
        M[Prefix Hash 索引]
        M --> E1[Hash: abc123<br/>节点: [N1, N3]<br/>长度: 4096]
        M --> E2[Hash: def456<br/>节点: [N2]<br/>长度: 8192]
        M --> E3[Hash: ghi789<br/>节点: [N1, N2, N4]<br/>长度: 2048]
    end
```

查询结果告诉 Conductor：
- 哪些节点缓存了请求的部分或全部前缀
- 各节点的缓存命中长度

#### 4.2.3 调度决策

Conductor 综合考虑以下因素做出调度决策：

**1. Cache 命中优化**

选择缓存命中长度最大的节点，可以最大程度减少 Prefill 计算量：

```
节省的计算量 ∝ cached_prefix_length²
```

**2. 负载均衡**

避免将请求都调度到同一节点：

```python
# 伪代码：调度得分计算
def compute_score(node, request):
    cache_hit_length = get_cache_hit_length(node, request.prefix_hash)
    cache_score = cache_hit_length / request.total_length

    load = get_node_load(node)
    load_score = 1.0 - load

    return alpha * cache_score + beta * load_score
```

**3. 网络拓扑**

优先选择与数据源网络距离近的节点：

```mermaid
graph TB
    subgraph "机架 1"
        N1[节点 1]
        N2[节点 2]
    end
    subgraph "机架 2"
        N3[节点 3]
        N4[节点 4]
    end

    N1 <-->|低延迟| N2
    N3 <-->|低延迟| N4
    N1 <-->|高延迟| N3
```

如果 KVCache 主要在机架 1 的节点上，优先选择同机架的 Prefill 节点。

### 4.3 阶段二：Prefill 处理

#### 4.3.1 KVCache 加载

Prefill 节点收到任务后，首先从 Mooncake Store 加载已缓存的 KVCache：

```mermaid
sequenceDiagram
    participant P as Prefill Node
    participant TE as Transfer Engine
    participant S1 as Store Node 1
    participant S2 as Store Node 2

    P->>TE: 请求加载 KVCache chunks [1, 2, 3, ...]

    par 并行传输
        TE->>S1: RDMA Read chunk 1, 3
        TE->>S2: RDMA Read chunk 2, 4
    end

    S1-->>TE: chunk 1, 3 数据
    S2-->>TE: chunk 2, 4 数据

    TE-->>P: KVCache 加载完成
```

**Transfer Engine 的并行传输机制**（来自 `rdma_transport.cpp`）：

```cpp
Status RdmaTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest>& entries) {

    // 将大传输分割成 slices
    const size_t kBlockSize = globalConfig().slice_size;  // 通常 64KB

    for (auto& request : entries) {
        for (uint64_t offset = 0; offset < request.length;
             offset += kBlockSize) {

            Slice* slice = getSliceCache().allocate();
            slice->source_addr = (char*)request.source + offset;
            slice->length = std::min(kBlockSize, request.length - offset);
            slice->opcode = request.opcode;
            slice->rdma.dest_addr = request.target_offset + offset;

            // 选择最优网卡
            int device_id = selectDevice(...);
            slices_to_post[context_list_[device_id]].push_back(slice);
        }
    }

    // 提交到各网卡并行执行
    for (auto& entry : slices_to_post) {
        entry.first->submitPostSend(entry.second);
    }
}
```

#### 4.3.2 增量 Prefill

只对未缓存的 tokens 执行 Prefill 计算：

```mermaid
graph LR
    subgraph "输入 Tokens"
        T1[Token 1]
        T2[Token 2]
        T3[...]
        T4[Token 4096]
        T5[Token 4097]
        T6[...]
        T7[Token 8000]
    end

    subgraph "处理方式"
        C[从 Cache 加载<br/>Token 1-4096]
        P[Prefill 计算<br/>Token 4097-8000]
    end

    T1 & T2 & T3 & T4 --> C
    T5 & T6 & T7 --> P

    style T1 fill:#90EE90
    style T2 fill:#90EE90
    style T3 fill:#90EE90
    style T4 fill:#90EE90
    style T5 fill:#FFB6C1
    style T6 fill:#FFB6C1
    style T7 fill:#FFB6C1
```

**增量 Prefill 的计算节省**：

假设总长度 8000 tokens，缓存命中 4096 tokens：
- 传统方式：需要计算 8000² ≈ 6400 万次注意力操作
- 增量方式：只需计算 (8000² - 4096²) ≈ 4720 万次注意力操作
- **节省：26% 计算量**

#### 4.3.3 KVCache 存储

Prefill 完成后，将新生成的 KVCache 存储到 Mooncake Store：

```cpp
// 伪代码：KVCache 存储流程
void store_kvcache(KVCache& cache, PrefixHash& hash) {
    // 1. 分割成 chunks
    auto chunks = cache.split_into_chunks(CHUNK_SIZE);

    // 2. 计算存储位置
    std::vector<NodeId> target_nodes = select_storage_nodes(chunks);

    // 3. 并行写入
    BatchID batch = engine.allocateBatchID(chunks.size());
    for (size_t i = 0; i < chunks.size(); i++) {
        TransferRequest req;
        req.opcode = WRITE;
        req.source = chunks[i].data();
        req.target_id = target_nodes[i];
        req.target_offset = get_chunk_offset(hash, i);
        req.length = chunks[i].size();
        requests.push_back(req);
    }
    engine.submitTransfer(batch, requests);

    // 4. 等待完成
    wait_for_completion(batch);

    // 5. 更新元数据
    update_prefix_metadata(hash, target_nodes);
}
```

### 4.4 阶段三：Decode 处理

#### 4.4.1 Decode 节点分配

Prefill 完成后，Conductor 选择 Decode 节点。选择策略考虑：

1. **KVCache 位置**：优先选择已有 KVCache 的节点（避免传输）
2. **当前负载**：选择有足够空闲 batch 槽位的节点
3. **TBT SLO**：确保选择的节点能满足 TBT 要求

```mermaid
graph TB
    subgraph "Decode 节点选择"
        C[Conductor] --> Q{KVCache 在哪？}
        Q -->|已在 Decode 节点| D1[直接使用该节点]
        Q -->|在 Store 中| D2[选择空闲节点 + 传输]
        Q -->|在 Prefill 节点| D3[考虑是否迁移]
    end
```

#### 4.4.2 KVCache 迁移（如需要）

如果 KVCache 不在目标 Decode 节点上，需要进行迁移：

```mermaid
sequenceDiagram
    participant C as Conductor
    participant P as Prefill Node
    participant D as Decode Node

    C->>P: 通知 Prefill 完成
    C->>D: 分配 Decode 任务

    alt KVCache 在 Prefill 节点
        P->>D: 直接传输 KVCache
    else KVCache 在 Store
        D->>Store: 加载 KVCache
    end

    D->>D: 开始 Decode
```

**传输与计算重叠**：

Mooncake 支持边传输边计算，进一步减少延迟：

```mermaid
gantt
    title KVCache 传输与 Decode 重叠
    dateFormat X
    axisFormat %s

    section 传输
    Chunk 1-4    :t1, 0, 10
    Chunk 5-8    :t2, 10, 20

    section Decode
    使用 Chunk 1-4 :d1, 10, 25
    使用 Chunk 5-8 :d2, 20, 35
```

#### 4.4.3 自回归生成

Decode 节点进入自回归生成循环：

```python
# 伪代码：Decode 循环
def decode_loop(kvcache, prompt_tokens, max_new_tokens):
    generated_tokens = []
    current_kvcache = kvcache

    for i in range(max_new_tokens):
        # 1. 获取最后一个 token
        last_token = generated_tokens[-1] if generated_tokens else prompt_tokens[-1]

        # 2. 前向计算 (使用 KVCache)
        logits = model.forward(last_token, current_kvcache)

        # 3. 采样下一个 token
        next_token = sample(logits, temperature, top_p)

        # 4. 更新 KVCache
        current_kvcache.append(next_token_kv)

        # 5. 流式返回
        yield next_token

        # 6. 检查停止条件
        if next_token == EOS_TOKEN:
            break

        generated_tokens.append(next_token)
```

**Continuous Batching**：

Decode 节点通常同时处理多个请求，使用 Continuous Batching 最大化 GPU 利用率：

```mermaid
graph TB
    subgraph "Iteration 1"
        B1[Batch: A, B, C, D]
    end

    subgraph "Iteration 2"
        B2[Batch: A, B, D, E]
        Note1[C 完成, E 加入]
    end

    subgraph "Iteration 3"
        B3[Batch: A, D, E, F]
        Note2[B 完成, F 加入]
    end

    B1 --> B2 --> B3
```

#### 4.4.4 KVCache 增量存储

Decode 过程中，新生成的 KVCache 可以选择性地存储到 Mooncake Store：

- **热数据**：高频访问的前缀，主动复制到多节点
- **冷数据**：低频访问的数据，可以延迟写入或丢弃

### 4.5 阶段四：完成与资源回收

#### 4.5.1 请求完成处理

当 Decode 完成（生成 EOS 或达到 max_tokens）：

```mermaid
sequenceDiagram
    participant D as Decode Node
    participant C as Conductor
    participant S as Mooncake Store
    participant Client as 客户端

    D->>Client: 发送结束标记
    D->>C: 通知请求完成
    C->>C: 更新统计信息

    alt 保留 KVCache
        D->>S: 提交 KVCache 存储
        S->>S: 更新驱逐策略 (LRU/SIEVE)
    else 丢弃 KVCache
        D->>D: 释放本地 KVCache 内存
    end
```

#### 4.5.2 缓存驱逐策略

从 `eviction_strategy.h` 可以看到 Mooncake 支持的驱逐策略：

**LRU (Least Recently Used)**：

```cpp
class LRUEvictionStrategy : public EvictionStrategy {
public:
    ErrorCode AddKey(const std::string& key) override {
        // 新 key 添加到队列头部
        all_key_list_.push_front(key);
        all_key_idx_map_[key] = all_key_list_.begin();
        return ErrorCode::OK;
    }

    ErrorCode UpdateKey(const std::string& key) override {
        // 访问时移动到头部
        auto it = all_key_idx_map_.find(key);
        if (it != all_key_idx_map_.end()) {
            all_key_list_.erase(it->second);
            all_key_list_.push_front(key);
            all_key_idx_map_[key] = all_key_list_.begin();
        }
        return ErrorCode::OK;
    }

    std::string EvictKey() override {
        // 驱逐尾部 (最久未使用)
        if (all_key_list_.empty()) return "";
        std::string evicted_key = all_key_list_.back();
        all_key_list_.pop_back();
        all_key_idx_map_.erase(evicted_key);
        return evicted_key;
    }
};
```

**FIFO (First In First Out)**：

```cpp
class FIFOEvictionStrategy : public EvictionStrategy {
public:
    ErrorCode AddKey(const std::string& key) override {
        // 新 key 添加到头部
        all_key_list_.push_front(key);
        return ErrorCode::OK;
    }

    ErrorCode UpdateKey(const std::string& key) override {
        // FIFO 不关心访问顺序
        return ErrorCode::OK;
    }

    std::string EvictKey() override {
        // 驱逐尾部 (最早进入)
        if (all_key_list_.empty()) return "";
        std::string evicted_key = all_key_list_.back();
        all_key_list_.pop_back();
        return evicted_key;
    }
};
```

### 4.6 异常处理与容错

#### 4.6.1 节点故障处理

```mermaid
graph TB
    subgraph "故障检测"
        H[心跳超时] --> D[节点被标记为 Down]
    end

    subgraph "Prefill 故障恢复"
        D --> P1{Prefill 是否完成?}
        P1 -->|否| P2[重新调度到其他节点]
        P1 -->|是| P3[继续 Decode 流程]
    end

    subgraph "Decode 故障恢复"
        D --> D1{KVCache 是否已存储?}
        D1 -->|是| D2[从 Store 恢复并重新调度]
        D1 -->|否| D3[从 Prefill 重新开始]
    end
```

#### 4.6.2 传输失败重试

Transfer Engine 内置了重试机制（来自 `transport.h`）：

```cpp
struct Slice {
    // ...
    struct {
        uint32_t retry_cnt;      // 当前重试次数
        uint32_t max_retry_cnt;  // 最大重试次数
    } rdma;

    void markFailed() {
        if (rdma.retry_cnt < rdma.max_retry_cnt) {
            // 重新提交
            status = PENDING;
            rdma.retry_cnt++;
            resubmit();
        } else {
            // 永久失败
            status = FAILED;
            task->failed_slice_count++;
        }
    }
};
```

### 4.7 性能关键路径分析

#### 4.7.1 延迟分解

一个典型请求的延迟分解：

```mermaid
pie title 请求延迟分解 (128K context)
    "调度决策" : 5
    "KVCache 加载" : 15
    "Prefill 计算" : 40
    "KVCache 存储" : 10
    "Decode 循环" : 30
```

#### 4.7.2 关键优化点

| 阶段 | 优化技术 | 效果 |
|------|----------|------|
| 调度 | 预计算 prefix hash | 减少决策延迟 |
| 加载 | 多网卡并行 | 2-8x 带宽提升 |
| Prefill | Prefix Caching | 减少 20-80% 计算 |
| 传输 | 流水线 | 隐藏传输延迟 |
| Decode | Continuous Batching | 提高 GPU 利用率 |

### 4.8 本章小结

本章详细分析了 Mooncake 中请求的完整处理流程：

1. **调度阶段**：Cache-aware 调度、负载均衡、拓扑感知
2. **Prefill 阶段**：KVCache 加载、增量计算、存储
3. **Decode 阶段**：KVCache 迁移、自回归生成、Continuous Batching
4. **完成阶段**：资源回收、缓存驱逐
5. **容错机制**：故障检测、重试、恢复

这种精心设计的流程确保了 Mooncake 能够在满足 SLO 要求的同时，最大化资源利用率和缓存命中率。

---

[上一篇：整体架构设计](01-system-architecture.md) | [目录](../README.md) | [下一篇：存储架构设计](../03-mooncake-store/01-storage-architecture.md)
