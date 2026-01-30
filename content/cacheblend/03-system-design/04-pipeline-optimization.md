---
title: "Pipeline 优化"
weight: 4
---

本文档介绍 CacheBlend 的流水线优化策略，通过将 KV Cache 加载与选择性重计算并行化，实现延迟隐藏，最大限度减少 TTFT（Time To First Token）。

---

## 3.5.1 流水线并行策略

CacheBlend 的一个关键优化是将 KV 加载与选择性重计算流水线化：

```mermaid
gantt
    title CacheBlend 流水线时序
    dateFormat X
    axisFormat %s

    section Layer 1
    加载 KV     :a1, 0, 16
    重计算      :a2, after a1, 3

    section Layer 2
    加载 KV     :b1, after a1, 16
    重计算      :b2, after b1, 3

    section Layer 3
    加载 KV     :c1, after b1, 16
    重计算      :c2, after c1, 3
```

## 3.5.2 两线程实现

在第 $i$ 层的部分 Prefill 中，使用两个线程来流水线化：
- **线程 1**：执行第 $i$ 层的计算（prefill_layer）
- **线程 2**：加载第 $i+1$ 层的 KV Cache（fetch_kv）

在 prefill_layer 之前调用 synchronize 以确保 Prefill 所需的 KV Cache 已加载到 GPU。

## 3.5.3 延迟隐藏效果

当加载延迟 >= 重计算延迟时：
- KV 重计算延迟被完全隐藏
- TTFT 不增加额外延迟

```mermaid
graph LR
    subgraph "延迟隐藏"
        A["无流水线<br>TTFT = T_load + T_recompute"]
        B["有流水线<br>TTFT = max(T_load, T_recompute)"]

        A -->|"流水线优化"| B
    end

    style B fill:#90EE90
```

## 3.5.4 系统完整流程

将所有组件整合在一起：

```mermaid
sequenceDiagram
    participant User as 用户
    participant LC as Loading Controller
    participant KS as KV Cache Manager
    participant F as Fusor
    participant LLM as LLM Engine

    User->>LC: 1. 提交问题 + 相关文本块列表
    LC->>KS: 2. 查询 KV Cache 存在性和位置
    KS->>LC: 3. 返回信息
    LC->>LC: 4. 计算理想重计算比例
    LC->>F: 5. 发送比例
    LC->>F: 6. 开始加载 KV Cache 到 GPU 队列

    loop 每一层
        F->>F: 7. 在队列中的 KV Cache 上重计算
    end

    F->>LLM: 8. 提供融合的 KV Cache
    LLM->>User: 9. 基于 KV Cache 生成答案
```

---

**上一步**: [KV Cache Store 与 Fusor](./03-kv-cache-store.md)

**下一步**: [第四部分 - 代码实现深度解析](../04-implementation/README.md)
