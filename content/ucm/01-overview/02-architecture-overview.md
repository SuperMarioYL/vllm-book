---
title: "UCM 架构概览"
weight: 2
---


> **阅读时间**: 约 15 分钟
> **前置要求**: [UCM 项目简介](./01-introduction.md)

---

## 概述

本文介绍 UCM 的整体架构设计，包括四大核心模块、数据流、扩展点等内容。

---

## 1. 四大支柱架构

UCM 采用模块化设计，由四大核心模块组成：

```mermaid
graph TB
    subgraph ucm["UCM 架构"]
        subgraph integration["集成层 - Integration"]
            I1["vLLM Connector"]
            I2["Monkey Patching"]
        end

        subgraph sparse["算法层 - Sparse"]
            S1["ESA / GSA"]
            S2["Blend / KVStar"]
            S3["RERoPE"]
        end

        subgraph store["存储层 - Store"]
            ST1["POSIX / Cache"]
            ST2["NFS / DS3FS"]
            ST3["Pipeline"]
        end

        subgraph shared["共享层 - Shared"]
            SH1["Transport"]
            SH2["Metrics"]
            SH3["Logging"]
        end

        integration --> sparse
        integration --> store
        sparse --> shared
        store --> shared
    end

    subgraph external["外部系统"]
        vLLM["vLLM v0.9.2"]
        Storage["外部存储"]
    end
    vLLM --> integration
    store --> Storage
```
### 1.1 模块职责
| 模块 | 目录 | 职责 |
|------|------|------|
| **集成层** | `ucm/integration/vllm/` | 与 vLLM 的接口对接、Monkey Patching |
| **算法层** | `ucm/sparse/` | 稀疏注意力算法实现 |
| **存储层** | `ucm/store/` | KV Cache 存储后端管理 |
| **共享层** | `ucm/shared/` | 传输、监控、日志等基础设施 |

---

## 2. 目录结构

```
ucm/
├── integration/                # 集成层
│   └── vllm/
│       ├── ucm_connector.py    # 主连接器实现
│       ├── blend_connector.py  # Blend 专用连接器
│       └── patch/              # Monkey Patching
│           ├── apply_patch.py
│           └── patch_funcs/
│               └── v092/       # vLLM v0.9.2 补丁
│
├── sparse/                     # 算法层
│   ├── base.py                 # 稀疏算法基类
│   ├── factory.py              # 算法工厂
│   ├── esa/                    # ESA 算法
│   ├── gsa/                    # GSA 算法
│   ├── gsa_on_device/          # GPU 端 GSA
│   ├── blend/                  # Blend 算法
│   ├── kvstar/                 # KVStar 算法
│   └── rerope/                 # RERoPE 算法
│
├── store/                      # 存储层
│   ├── ucmstore.py             # 存储基类
│   ├── factory.py              # 存储工厂
│   ├── posix/                  # 本地文件存储
│   ├── cache/                  # 内存缓存
│   ├── pipeline/               # Pipeline 组合
│   ├── nfsstore/               # NFS 存储
│   ├── ds3fs/                  # S3 存储
│   └── mooncakestore/          # 云存储
│
├── shared/                     # 共享层
│   ├── trans/                  # 设备传输
│   ├── metrics/                # 监控指标
│   └── infra/                  # 基础工具
│
└── logger.py                   # 日志配置
```
---
## 3. 核心组件关系

### 3.1 类图概览

```mermaid
classDiagram
    class KVConnectorBase {
        <<vLLM接口>>
        +get_num_new_matched_tokens()
        +build_connector_meta()
        +start_load_kv()
        +wait_for_save()
    }

    class UCMDirectConnector {
        -store: UcmKVStoreBase
        -sparse: UcmSparseBase
        -hasher: RequestHasher
        +get_num_new_matched_tokens()
        +start_load_kv()
    }

    class UcmKVStoreBase {
        <<abstract>>
        +lookup(block_ids)
        +load(block_ids, tensor)
        +dump(block_ids, tensor)
        +wait(task)
    }

    class UcmSparseBase {
        <<abstract>>
        +request_begin()
        +build_sparse_meta()
        +attention_begin()
        +attention_finished()
    }

    class UcmConnectorFactory {
        +create_connector(name, config)
    }

    class UcmSparseFactory {
        +create_sparse_method(name, config)
    }

    KVConnectorBase <|-- UCMDirectConnector
    UCMDirectConnector --> UcmKVStoreBase
    UCMDirectConnector --> UcmSparseBase
    UcmConnectorFactory --> UcmKVStoreBase
    UcmSparseFactory --> UcmSparseBase
```

### 3.2 组件交互

```mermaid
sequenceDiagram
    participant vLLM as vLLM Scheduler
    participant Conn as UCMConnector
    participant Store as UcmKVStore
    participant Sparse as UcmSparse

    Note over vLLM,Sparse: 请求到达阶段

    vLLM->>Conn: get_num_new_matched_tokens()
    Conn->>Conn: 生成 Block Hash
    Conn->>Store: lookup(block_ids)
    Store-->>Conn: 命中结果
    Conn-->>vLLM: 返回命中数量

    Note over vLLM,Sparse: Worker 执行阶段

    vLLM->>Conn: start_load_kv()
    Conn->>Store: load(block_ids, tensor)

    vLLM->>Sparse: attention_begin()
    Sparse-->>vLLM: 稀疏 KV 数据

    vLLM->>Conn: wait_for_save()
    Conn->>Store: dump(block_ids, tensor)
```

---
## 4. 数据流
### 4.1 完整请求处理流程
```mermaid
flowchart TB
    subgraph phase1["阶段 1 - 请求到达"]
        A1["用户请求"] --> A2["Token 化"]
        A2 --> A3["生成 Block Hash"]
        A3 --> A4["查询 KV 命中"]
    end
    subgraph phase2["阶段 2 - 调度决策"]
        B1["计算需加载的 Block"]
        B2["计算需计算的 Block"]
        B3["构建调度元数据"]
        A4 --> B1
        B1 --> B2
        B2 --> B3
    end

    subgraph phase3["阶段 3 - KV 加载"]
        C1["从存储加载 KV"]
        C2["传输到 GPU"]
        B3 --> C1
        C1 --> C2
    end
    subgraph phase4["阶段 4 - 模型计算"]
        D1["Prefill / Decode"]
        D2["稀疏注意力"]
        D3["生成 Token"]
        C2 --> D1
        D1 --> D2
        D2 --> D3
    end

    subgraph phase5["阶段 5 - KV 保存"]
        E1["新 KV 传输到 CPU"]
        E2["持久化到存储"]
        D3 --> E1
        E1 --> E2
    end
```

### 4.2 KV Cache 数据路径

```mermaid
graph LR
    subgraph compute["计算路径"]
        GPU["GPU HBM<br/>注意力计算"]
    end

    subgraph cache["缓存路径"]
        CPU["CPU Pinned Memory<br/>Cache Store"]
    end

    subgraph persist["持久化路径"]
        Local["本地存储<br/>POSIX Store"]
        Remote["远程存储<br/>NFS/DS3FS"]
    end
    GPU <--> |"加载/保存"| CPU
    CPU <--> |"淘汰/预取"| Local
    Local <--> |"归档/恢复"| Remote
```

---

## 5. 扩展点设计

### 5.1 存储后端扩展

添加新的存储后端只需：

1. 实现 `UcmKVStoreBase` 接口
2. 在工厂中注册
```python
class MyCustomStore(UcmKVStoreBase):
    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        # 实现查找逻辑
        pass
    def load(self, block_ids, tensor) -> Task:
        # 实现加载逻辑
        pass
    def dump(self, block_ids, tensor) -> Task:
        # 实现保存逻辑
        pass
# 注册到工厂
UcmConnectorFactory.register_connector(
    "MyCustomStore",
    "my_module.custom_store",
    "MyCustomStore"
)
```
### 5.2 稀疏算法扩展
添加新的稀疏算法只需：
1. 实现 `UcmSparseBase` 接口
2. 在工厂中注册

```python
# 自定义稀疏算法示例
class MySparsAlgorithm(UcmSparseBase):
    def build_sparse_meta(self, scheduler_output, ...):
        # 构建稀疏元数据
        pass
    def attention_begin(self, layer_idx, ...):
        # 注意力前处理
        pass
    def attention_finished(self, layer_idx, ...):
        # 注意力后处理
        pass
# 注册到工厂
UcmSparseFactory.register_sparse_method(
    "MySparse",
    "my_module.sparse",
    "MySparsAlgorithm"
)
```
---
## 6. 配置系统

### 6.1 配置文件结构

```yaml

ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/data/ucm_cache"
      buffer_number: 2048

ucm_sparse_config:
  ESA:
    sparse_ratio: 0.3
    local_window_sz: 2
    retrieval_stride: 5

metrics_config_path: "./metrics_config.yaml"

load_only_first_rank: false
use_layerwise: true
```

### 6.2 配置优先级

```
高优先级 ──────────────────────────────► 低优先级

环境变量 > kv_connector_extra_config > YAML 文件 > 默认值
```

---
## 7. 监控与可观测性
### 7.1 核心指标
| 指标 | 类型 | 说明 |
|------|------|------|
| `ucm_lookup_hit_rate` | Gauge | KV 查询命中率 |
| `ucm_load_duration_ms` | Histogram | 加载耗时分布 |
| `ucm_load_speed_gbps` | Gauge | 加载速度 |
| `ucm_save_duration_ms` | Histogram | 保存耗时分布 |
| `ucm_save_speed_gbps` | Gauge | 保存速度 |
### 7.2 日志配置
```bash
export UNIFIED_CACHE_LOG_LEVEL=DEBUG  # DEBUG/INFO/WARNING/ERROR
```
---
## 8. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| vLLM 集成方式 | Monkey Patching | 无需修改源码，易于升级 |
| Block 标识 | MD5 哈希 | 平衡速度和碰撞率 |
| 存储抽象 | 异步任务模型 | 支持计算传输重叠 |
| 稀疏接口 | 生命周期钩子 | 灵活支持各类算法 |
