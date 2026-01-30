# SGLang 原理机制教程

本教程面向深度学习开发者，系统性地讲解 SGLang 的内部原理和实现机制。

## 教程结构

本教程共 8 个模块、29 篇文档，涵盖从入门到高级的完整学习路径。

```
tutorials/
├── 01-getting-started/     # 入门篇
├── 02-core-architecture/   # 核心架构篇
├── 03-memory-cache/        # 内存与缓存篇
├── 04-batching-scheduling/ # 批处理与调度篇
├── 05-performance/         # 性能优化篇
├── 06-distributed/         # 分布式与并行篇
├── 07-advanced-features/   # 高级特性篇
└── 08-debugging/           # 实战调试篇
```

---

## 模块一：入门篇

| 文档 | 内容 |
|------|------|
| [01-introduction](01-getting-started/01-introduction.md) | SGLang 概述、核心特性、与 vLLM 对比 |
| [02-quick-start](01-getting-started/02-quick-start.md) | 安装、启动服务、发送请求 |
| [03-architecture-overview](01-getting-started/03-architecture-overview.md) | 三进程架构、数据流、组件职责 |

---

## 模块二：核心架构篇

| 文档 | 内容 |
|------|------|
| [04-request-lifecycle](02-core-architecture/04-request-lifecycle.md) | 请求生命周期、ZMQ 通信、数据流转 |
| [05-tokenizer-manager](02-core-architecture/05-tokenizer-manager.md) | TokenizerManager、ReqState、异步处理 |
| [06-scheduler-deep-dive](02-core-architecture/06-scheduler-deep-dive.md) | 调度器核心逻辑、事件循环、批处理 |
| [07-detokenizer](02-core-architecture/07-detokenizer.md) | 增量解码、流式输出机制 |
| [08-model-runner](02-core-architecture/08-model-runner.md) | ModelRunner、ForwardBatch、前向执行 |

---

## 模块三：内存与缓存篇

| 文档 | 内容 |
|------|------|
| [09-kv-cache-basics](03-memory-cache/09-kv-cache-basics.md) | KV Cache 原理、内存占用分析 |
| [10-radix-attention](03-memory-cache/10-radix-attention.md) | RadixAttention、Radix Tree、前缀匹配 |
| [11-memory-pool](03-memory-cache/11-memory-pool.md) | 内存池设计、Paged Attention 实现 |
| [12-eviction-policy](03-memory-cache/12-eviction-policy.md) | 缓存淘汰策略、内存回收 |

---

## 模块四：批处理与调度篇

| 文档 | 内容 |
|------|------|
| [13-continuous-batching](04-batching-scheduling/13-continuous-batching.md) | 连续批处理原理、吞吐量提升 |
| [14-chunked-prefill](04-batching-scheduling/14-chunked-prefill.md) | Chunked Prefill、Mixed Mode |
| [15-schedule-policy](04-batching-scheduling/15-schedule-policy.md) | 调度策略 LPM/FCFS/DFS-Weight |
| [16-forward-modes](04-batching-scheduling/16-forward-modes.md) | ForwardMode 详解 |

---

## 模块五：性能优化篇

| 文档 | 内容 |
|------|------|
| [17-cuda-graph](05-performance/17-cuda-graph.md) | CUDA Graph 原理、捕获与回放 |
| [18-torch-compile](05-performance/18-torch-compile.md) | torch.compile 集成、内核融合 |
| [19-quantization](05-performance/19-quantization.md) | 量化支持 FP8/INT4/AWQ/GPTQ |
| [20-speculative-decoding](05-performance/20-speculative-decoding.md) | 推测解码 EAGLE/NGRAM |

---

## 模块六：分布式与并行篇

| 文档 | 内容 |
|------|------|
| [21-tensor-parallelism](06-distributed/21-tensor-parallelism.md) | 张量并行原理与实现 |
| [22-pipeline-parallelism](06-distributed/22-pipeline-parallelism.md) | 流水线并行、气泡优化 |
| [23-expert-parallelism](06-distributed/23-expert-parallelism.md) | 专家并行 MoE、DeepSeek 优化 |

---

## 模块七：高级特性篇

| 文档 | 内容 |
|------|------|
| [24-structured-output](07-advanced-features/24-structured-output.md) | JSON Schema、正则约束 |
| [25-lora-support](07-advanced-features/25-lora-support.md) | LoRA 多适配器、动态加载 |
| [26-multimodal](07-advanced-features/26-multimodal.md) | 多模态 VLM、图像/视频处理 |

---

## 模块八：实战调试篇

| 文档 | 内容 |
|------|------|
| [27-debugging-guide](08-debugging/27-debugging-guide.md) | 调试技巧、关键断点 |
| [28-profiling](08-debugging/28-profiling.md) | 性能分析、瓶颈定位 |
| [29-troubleshooting](08-debugging/29-troubleshooting.md) | 常见问题、调优 checklist |

---

## 学习路径建议

### 初学者路径

```
入门篇 (01-03) → 核心架构篇 (04-08) → 实战调试篇 (27-29)
```

### 性能优化路径

```
批处理与调度篇 (13-16) → 性能优化篇 (17-20) → 实战调试篇 (28)
```

### 分布式部署路径

```
核心架构篇 (04-08) → 分布式与并行篇 (21-23)
```

### 高级功能路径

```
内存与缓存篇 (09-12) → 高级特性篇 (24-26)
```

---

## 前置要求

- Python 3.9+
- PyTorch 2.0+
- CUDA 12.0+
- 了解 Transformer 架构
- 熟悉 LLM 推理概念

---

## 关键源码文件

| 组件 | 文件路径 |
|------|---------|
| HTTP Server | `python/sglang/srt/entrypoints/http_server.py` |
| Engine | `python/sglang/srt/entrypoints/engine.py` |
| Scheduler | `python/sglang/srt/managers/scheduler.py` |
| Tokenizer Manager | `python/sglang/srt/managers/tokenizer_manager.py` |
| Detokenizer | `python/sglang/srt/managers/detokenizer_manager.py` |
| Model Runner | `python/sglang/srt/model_executor/model_runner.py` |
| CUDA Graph | `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| Radix Cache | `python/sglang/srt/mem_cache/radix_cache.py` |
| Memory Pool | `python/sglang/srt/mem_cache/memory_pool.py` |
| Forward Batch | `python/sglang/srt/model_executor/forward_batch_info.py` |
| Schedule Policy | `python/sglang/srt/managers/schedule_policy.py` |

---

## 参考资源

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang 官方文档](https://sgl-project.github.io/)
- [SGLang Paper](https://arxiv.org/abs/2312.07104)

---

## 贡献指南

欢迎提交 Issue 和 PR 来改进本教程：

1. 修复错误或过时内容
2. 补充更多示例代码
3. 添加新的主题章节
4. 改进图表和说明

