---
title: "GPU/CPU 内存层级与数据传输"
weight: 4
---


> **阅读时间**: 约 15 分钟
> **前置要求**: [vLLM 架构概览](./03-vllm-architecture.md)

---

## 概述

理解硬件内存层级和数据传输特性是优化 LLM 推理的关键。本文介绍 GPU/CPU 内存架构以及 UCM 如何利用这些特性进行优化。

---

## 1. 内存层级金字塔

### 1.1 层级结构

```mermaid
graph TB
    subgraph pyramid["内存层级金字塔"]
        L1["GPU 寄存器/L1 Cache<br/>最快 ~1ns | 最小 ~KB"]
        L2["GPU L2 Cache<br/>快 ~10ns | 小 ~MB"]
        HBM["GPU HBM (显存)<br/>中 ~300ns | 中 40-80GB"]
        CPU["CPU DRAM (内存)<br/>慢 ~100ns | 大 100GB-TB"]
        SSD["NVMe SSD<br/>很慢 ~10μs | 很大 TB级"]
        NET["网络存储<br/>最慢 ~ms | 最大 PB级"]

        L1 --> L2
        L2 --> HBM
        HBM --> CPU
        CPU --> SSD
        SSD --> NET
    end
    subgraph properties["特性"]
        P1["越往上: 越快、越贵、越小"]
        P2["越往下: 越慢、越便宜、越大"]
    end
```
### 1.2 各层级参数对比
| 存储层级 | 延迟 | 带宽 | 典型容量 | 每 GB 成本 |
|----------|------|------|----------|------------|
| GPU 寄存器 | ~1 ns | ~TB/s | ~KB | - |
| GPU L1 Cache | ~5 ns | ~10 TB/s | ~128 KB | - |
| GPU L2 Cache | ~30 ns | ~5 TB/s | ~50 MB | - |
| GPU HBM | ~300 ns | ~2 TB/s | 40-80 GB | $$$ |
| CPU DRAM | ~100 ns | ~200 GB/s | 100 GB - 1 TB | $$ |
| NVMe SSD | ~10 μs | ~7 GB/s | 1-30 TB | $ |
| 网络存储 | ~1 ms | ~10 GB/s | PB 级 | ¢ |
---
## 2. GPU 显存（HBM）

### 2.1 HBM 特性

HBM (High Bandwidth Memory) 是现代 GPU 的主要内存：

```mermaid
graph TB
    subgraph hbm["GPU HBM 特性"]
        subgraph advantages["优势"]
            A1["高带宽<br/>A100: 2 TB/s<br/>H100: 3.35 TB/s"]
            A2["低延迟<br/>~300ns"]
            A3["直接访问<br/>GPU 计算单元直接访问"]
        end

        subgraph limitations["限制"]
            L1["容量有限<br/>40-80 GB"]
            L2["成本高<br/>占 GPU 成本大比例"]
            L3["独占<br/>每块 GPU 独立"]
        end
    end
```

### 2.2 HBM 使用分配

以 A100 80GB 为例：

```mermaid
pie title "A100 80GB HBM 使用分配（Llama-70B 示例）"
    "模型权重" : 35
    "KV Cache" : 30
    "激活值" : 10
    "CUDA 运行时" : 3
    "预留/碎片" : 2
```

**关键洞察**: KV Cache 占用了大量显存，是优化的重点目标。

---
## 3. CPU 内存与 Pinned Memory
### 3.1 普通内存 vs Pinned Memory
```mermaid
graph TB
    subgraph normal["普通内存（Pageable）"]
        N1["可以被操作系统换出"]
        N2["GPU 传输需要额外复制"]
        N3["传输路径长"]
    end
    subgraph pinned["Pinned Memory（固定内存）"]
        P1["锁定在物理内存"]
        P2["DMA 直接访问"]
        P3["传输速度更快"]
    end
    subgraph comparison["传输对比"]
        C1["普通内存 → GPU<br/>CPU→临时缓冲→GPU<br/>~10 GB/s"]
        C2["Pinned Memory → GPU<br/>CPU→GPU 直接DMA<br/>~25 GB/s"]
    end

    normal --> C1
    pinned --> C2
```

### 3.2 UCM 的 Pinned Memory 使用

UCM 使用 Pinned Memory 缓冲池加速数据传输：

```python
# UCM Pinned Memory 缓冲池示意
class PinnedMemoryPool:
    def __init__(self, buffer_count=1024, buffer_size=4096):
        # 预分配固定数量的 Pinned Memory 缓冲区
        self.buffers = [
            torch.empty(buffer_size, dtype=torch.uint8, pin_memory=True)
            for _ in range(buffer_count)
        ]

    def get_buffer(self):
        # 从池中获取可用缓冲区
        return self.free_buffers.pop()

    def return_buffer(self, buffer):
        # 归还缓冲区到池
        self.free_buffers.append(buffer)
```
---
## 4. 数据传输路径

### 4.1 PCIe 传输

CPU 和 GPU 之间通过 PCIe 总线传输数据：

```mermaid
graph LR
    subgraph cpu["CPU 侧"]
        DRAM["CPU DRAM"]
        PCIE_C["PCIe Controller"]
    end

    subgraph pcie["PCIe 总线"]
        BUS["PCIe Gen4/5<br/>32/64 GB/s"]
    end

    subgraph gpu["GPU 侧"]
        PCIE_G["PCIe Controller"]
        HBM["GPU HBM"]
    end

    DRAM --> PCIE_C
    PCIE_C --> BUS
    BUS --> PCIE_G
    PCIE_G --> HBM
```

### 4.2 NVLink（多 GPU）

NVLink 提供 GPU 之间的高速直连：

```mermaid
graph TB
    subgraph nvlink["NVLink 连接"]
        GPU1["GPU 0"] <--> |"NVLink<br/>600 GB/s"| GPU2["GPU 1"]
        GPU2 <--> |"NVLink"| GPU3["GPU 2"]
        GPU3 <--> |"NVLink"| GPU4["GPU 3"]
        GPU4 <--> |"NVLink"| GPU1
    end

    subgraph pcie["对比 PCIe"]
        P["PCIe Gen4<br/>32 GB/s"]
    end
```

### 4.3 传输带宽对比

| 传输路径 | 带宽 | 延迟 |
|----------|------|------|
| GPU 内部 (HBM) | ~2000 GB/s | ~300 ns |
| GPU-GPU (NVLink) | ~600 GB/s | ~1 μs |
| GPU-CPU (PCIe 4.0) | ~32 GB/s | ~10 μs |
| CPU-SSD (NVMe) | ~7 GB/s | ~10 μs |
| CPU-Network | ~10 GB/s | ~1 ms |

---
## 5. 异步传输与重叠
### 5.1 同步 vs 异步传输
```mermaid
sequenceDiagram
    participant CPU
    participant GPU

    Note over CPU,GPU: 同步传输 - 阻塞等待
    CPU->>GPU: 发起传输
    CPU->>CPU: 等待完成...
    GPU-->>CPU: 传输完成
    CPU->>CPU: 继续执行

    Note over CPU,GPU: 异步传输 - 并行执行
    CPU->>GPU: 发起传输（非阻塞）
    CPU->>CPU: 继续其他工作
    GPU-->>CPU: 传输完成通知
```

### 5.2 计算与传输重叠

UCM 利用异步传输实现计算与数据传输的重叠：

```mermaid
gantt
    title 计算与传输重叠示意
    dateFormat X
    axisFormat %s

    section 无重叠
    加载 KV :load1, 0, 3
    计算 Attention :compute1, 3, 5
    保存 KV :save1, 5, 8
    section 有重叠
    加载 KV (Layer N+1) :load2, 0, 3
    计算 Attention (Layer N) :compute2, 0, 2
    保存 KV (Layer N-1) :save2, 0, 2
```
### 5.3 CUDA Streams
CUDA Streams 用于管理异步操作：
```python
class AsyncTransfer:
    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
        self.transfer_stream = torch.cuda.Stream()
    def overlapped_execution(self, kv_data, compute_fn):
        # 在 transfer stream 上异步加载
        with torch.cuda.stream(self.transfer_stream):
            kv_gpu = kv_data.to('cuda', non_blocking=True)

        # 在 compute stream 上执行计算
        with torch.cuda.stream(self.compute_stream):
            # 等待传输完成
            self.compute_stream.wait_stream(self.transfer_stream)
            result = compute_fn(kv_gpu)

        return result
```
---
## 6. UCM 多级存储架构

### 6.1 存储层级设计

UCM 利用多级存储层级优化 KV Cache 管理：

```mermaid
graph TB
    subgraph ucm["UCM 多级存储"]
        subgraph l1["L1 - GPU HBM"]
            H1["热点 KV Cache<br/>最常访问的 Block"]
        end

        subgraph l2["L2 - CPU Pinned Memory"]
            H2["温数据<br/>Cache Store 缓存"]
        end

        subgraph l3["L3 - 本地存储"]
            H3["持久化 KV<br/>POSIX Store"]
        end

        subgraph l4["L4 - 分布式存储"]
            H4["共享 KV<br/>NFS/DS3FS/Mooncake"]
        end

        l1 --> |"淘汰"| l2
        l2 --> |"淘汰"| l3
        l3 --> |"归档"| l4

        l4 --> |"预取"| l3
        l3 --> |"加载"| l2
        l2 --> |"热身"| l1
    end
```

### 6.2 Pipeline Store 组合

UCM 支持灵活的存储层级组合：

```
配置示例:

1. Cache|Posix - 本地高速缓存
   HBM ↔ Pinned Memory Cache ↔ 本地 SSD

2. Cache|NFS - 分布式缓存
   HBM ↔ Pinned Memory Cache ↔ NFS 共享存储

3. Cache|DS3FS - 云存储缓存
   HBM ↔ Pinned Memory Cache ↔ S3 对象存储
```

---
## 7. 传输优化技术
### 7.1 批量传输
将多个小传输合并为大传输：
```mermaid
graph TB
    subgraph nobatch["无批量 - 多次小传输"]
        T1["传输 Block 1<br/>4KB"] --> T2["传输 Block 2<br/>4KB"]
        T2 --> T3["传输 Block 3<br/>4KB"]
        T3 --> T4["传输 Block 4<br/>4KB"]
    end
    subgraph batch["批量 - 一次大传输"]
        B1["传输 Block 1-4<br/>16KB 合并传输"]
    end
    nobatch --> |"4x 启动开销"| Result1["总时间长"]
    batch --> |"1x 启动开销"| Result2["总时间短"]
```
### 7.2 预取（Prefetch）
提前加载即将使用的数据：
```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant Prefetch as Prefetch Engine
    participant Store as Storage
    participant Worker as Worker

    Sched->>Prefetch: 预测下一批请求
    Prefetch->>Store: 提前加载 KV
    Store-->>Prefetch: KV 数据
    Prefetch->>Prefetch: 缓存到 GPU
    Note over Worker: 稍后...
    Worker->>Prefetch: 请求 KV
    Prefetch-->>Worker: 立即返回（已预取）
```
### 7.3 零拷贝（Zero-Copy）
减少不必要的数据复制：
```mermaid
graph TB
    subgraph copy["有拷贝"]
        S1["存储"] --> |"读取"| B1["缓冲区 1"]
        B1 --> |"复制"| B2["缓冲区 2"]
        B2 --> |"传输"| G1["GPU"]
    end
    subgraph zerocopy["零拷贝"]
        S2["存储"] --> |"直接传输"| G2["GPU"]
    end
```

---

## 8. 关键概念总结

| 概念 | 说明 | UCM 应用 |
|------|------|----------|
| HBM | GPU 高带宽显存 | 热点 KV 存储 |
| Pinned Memory | CPU 固定内存 | Cache Store 缓冲池 |
| PCIe | CPU-GPU 传输总线 | 决定传输带宽上限 |
| 异步传输 | 非阻塞数据传输 | 计算传输重叠 |
| CUDA Streams | 异步操作管理 | 并行调度 |
| 预取 | 提前加载数据 | Prefetch Engine |
| 多级存储 | 分层存储架构 | Pipeline Store |

---

## 延伸阅读

- [CUDA Programming Guide - Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [Understanding GPU Memory](https://developer.nvidia.com/blog/understanding-tensor-core-performance-with-nsight-compute/)