---
title: "KV 融合算法"
weight: 3
---

本文档详细介绍 CacheBlend 中 KV Cache 融合算法和层间传播算法的原理与实现。这两个算法协同工作，确保选择性重计算的结果正确传播到后续层。

---

## 5.3 KV Cache 融合算法

### 5.3.1 算法描述

在 status=2 的层中，将新计算的 HKVD token 的 KV 更新到旧 KV Cache 中。

### 5.3.2 核心代码

```python
# 文件: vllm_blend/vllm/attention/backends/xformers.py
# 位置: XFormersImpl.forward() 第 240-245 行

if status in [2]:
    # 获取 HKVD 索引
    imp_indices = cache_fuse_metadata["imp_indices"]

    # 核心：部分更新
    # 只更新 HKVD token 对应位置的 KV
    key_old[imp_indices] = key      # key 只包含 HKVD token
    value_old[imp_indices] = value  # value 只包含 HKVD token

    # 使用融合后的 KV 进行后续计算
    key = key_old
    value = value_old
```

### 5.3.3 融合过程图示

```mermaid
graph LR
    subgraph fusion["KV 融合过程"]
        OldKV["旧 KV Cache<br>[全部 token]"] --> Update["部分更新"]
        NewKV["新 KV<br>[仅 HKVD]"] --> Update
        Idx["imp_indices"] --> Update
        Update --> FusedKV["融合 KV Cache<br>[更新后的全部 token]"]
    end
```

### 5.3.4 内存效率分析

这种部分更新策略非常高效：
- 只需要存储 HKVD token 的新 KV（约 15%）
- 直接在 old_kv 上原地更新
- 不需要额外的大内存分配

---

## 5.4 层间传播算法

### 5.4.1 残差连接处理

在 Check 层（status=1）之后，残差连接也需要过滤：

```python
# 文件: vllm_blend/vllm/model_executor/models/llama.py
# 位置: LlamaDecoderLayer.forward() 第 264-265 行

if status == 1:
    # 过滤残差，只保留 HKVD token 对应的部分
    residual = residual[cache_fuse_metadata["imp_indices"]]
```

### 5.4.2 位置编码过滤

在 LlamaModel 中，位置编码也需要过滤：

```python
# 文件: vllm_blend/vllm/model_executor/models/llama.py
# 位置: LlamaModel.forward() 第 373-376 行

if temp_status == 1:
    # 过滤位置编码，只保留 HKVD token 的位置
    positions = positions[self.cache_fuse_metadata["imp_indices"]]
```

### 5.4.3 层间数据一致性

```mermaid
graph TB
    subgraph propagation["层间数据传播"]
        L1["Layer 1 (Check)"]
        L2["Layer 2+"]

        subgraph output["Check 层输出"]
            H1["hidden[imp_indices]"]
            R1["residual[imp_indices]"]
            P1["positions[imp_indices]"]
        end

        subgraph input["后续层输入"]
            H2["hidden"]
            R2["residual"]
            P2["positions"]
        end

        L1 --> H1
        L1 --> R1
        L1 --> P1

        H1 --> H2
        R1 --> R2
        P1 --> P2

        H2 --> L2
        R2 --> L2
        P2 --> L2
    end
```

### 5.4.4 完整的层间传播流程

```mermaid
sequenceDiagram
    participant L0 as Layer 0
    participant L1 as Layer 1 (Check)
    participant L2 as Layer 2
    participant LN as Layer N

    Note over L0,LN: 数据形状: [seq_len, hidden_size]

    L0->>L0: 处理所有 token
    L0->>L1: hidden[seq_len], residual[seq_len]

    Note over L1: HKVD 选择: imp_indices
    L1->>L1: 过滤 hidden, residual, positions
    Note over L1: 数据形状: [len(imp_indices), hidden_size]

    L1->>L2: hidden[HKVD], residual[HKVD]

    Note over L2: 部分更新 old_kv
    L2->>L2: old_kv[imp_indices] = new_kv
    L2->>L2: 使用融合 KV 计算注意力

    L2->>LN: hidden[HKVD], residual[HKVD]
    Note over LN: 继续部分更新和计算
```

---

## 下一步

- [实验设置](../06-experiments/01-setup.md) - 了解 CacheBlend 的实验评估方法
