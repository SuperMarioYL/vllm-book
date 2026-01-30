---
title: "整体架构"
weight: 1
---

本文档介绍 CacheBlend 系统的整体架构设计，包括核心组件、与 vLLM 的集成方式以及数据流与控制流的详细说明。

---

## 3.1.1 CacheBlend 系统组件

CacheBlend 系统由三个主要组件组成：

```mermaid
graph TB
    subgraph "CacheBlend 系统架构"
        User["用户查询"] --> Retriever["检索器"]
        Retriever --> Controller["Loading Controller<br>加载控制器"]

        subgraph "KV Cache Store"
            CPU["CPU RAM"]
            SSD["SSD"]
            Disk["Slower Disks"]
        end

        Controller --> Store["KV Cache Store<br>KV缓存存储"]
        Store --> CPU
        Store --> SSD
        Store --> Disk

        Controller --> Fusor["Fusor<br>缓存融合器"]
        Store --> Fusor

        Fusor --> LLM["LLM 推理引擎"]
        LLM --> Response["生成响应"]
    end

    style Controller fill:#87CEEB
    style Fusor fill:#90EE90
    style Store fill:#FFD700
```

## 3.1.2 与 vLLM 的集成

CacheBlend 在 vLLM 之上实现，约 3000 行 Python 代码。集成点包括：

```mermaid
graph TB
    subgraph "vLLM + CacheBlend 集成"
        subgraph "vLLM 核心"
            Engine["LLM Engine"]
            Executor["Model Executor"]
            Runner["Model Runner"]
        end

        subgraph "CacheBlend 扩展"
            Meta["cache_fuse_metadata"]
            OldKV["old_kvs"]
            Status["temp_status"]
        end

        subgraph "修改的模块"
            LlamaModel["LlamaModel"]
            LlamaAttn["LlamaAttention"]
            XFormers["XFormers Backend"]
        end

        Engine --> Executor
        Executor --> Runner
        Runner --> LlamaModel
        LlamaModel --> Meta
        LlamaModel --> OldKV
        LlamaModel --> Status
        LlamaAttn --> XFormers
    end
```

## 3.1.3 数据流与控制流

```mermaid
sequenceDiagram
    participant U as 用户
    participant R as 检索器
    participant LC as Loading Controller
    participant KS as KV Cache Store
    participant F as Fusor
    participant LLM as LLM Engine

    U->>R: 1. 提交查询
    R->>LC: 2. 返回相关文本块列表
    LC->>KS: 3. 查询 KV Cache 是否存在
    KS->>LC: 4. 返回 KV Cache 位置信息
    LC->>LC: 5. 计算理想重计算比例
    LC->>F: 6. 发送重计算比例
    LC->>KS: 7. 开始加载 KV Cache 到 GPU

    loop 每一层
        KS->>F: 8. 加载第 i 层 KV
        F->>F: 9. 执行选择性重计算
    end

    F->>LLM: 10. 提供融合的 KV Cache
    LLM->>U: 11. 生成并返回响应
```

---

**下一步**: [Loading Controller（加载控制器）](./02-loading-controller.md)
