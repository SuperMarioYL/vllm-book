---
title: "代码文件索引"
weight: 2
---

# 代码文件索引（Code Map）

本文档提供 vLLM 代码库的关键文件索引，帮助读者快速定位感兴趣的代码。

---

## 代码目录结构概览

```
vllm/
├── entrypoints/           # 入口点
│   ├── llm.py             # Python API 入口
│   ├── cli/               # 命令行入口
│   └── openai/            # OpenAI 兼容 API
│
├── v1/                    # V1 版本核心实现
│   ├── engine/            # 引擎相关
│   ├── core/              # 核心调度和内存管理
│   ├── worker/            # Worker 执行
│   ├── attention/         # 注意力实现
│   ├── sample/            # 采样器
│   └── spec_decode/       # 投机解码
│
├── model_executor/        # 模型执行
│   ├── models/            # 模型实现
│   └── layers/            # 层实现和量化
│
├── distributed/           # 分布式通信
│
├── config/                # 配置管理
│
└── csrc/                  # CUDA 内核
    └── attention/         # Attention CUDA 内核
```

---

## 入口点（Entry Points）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/entrypoints/llm.py` | Python API 入口 | `LLM`, `LLM.generate()`, `LLM.chat()` |
| `vllm/entrypoints/cli/main.py` | CLI 入口 | `serve`, `bench` 命令 |
| `vllm/entrypoints/openai/api_server.py` | OpenAI API 服务 | API 端点定义 |
| `vllm/engine/arg_utils.py` | 参数解析 | `EngineArgs`, `create_engine_config()` |

---

## V1 引擎（Engine）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/engine/llm_engine.py` | LLM 引擎入口 | `LLMEngine` |
| `vllm/v1/engine/core.py` | 引擎核心 | `EngineCore`, `EngineCore.step()` |
| `vllm/v1/engine/processor.py` | 输入/输出处理 | `InputProcessor`, `OutputProcessor` |
| `vllm/v1/engine/async_llm.py` | 异步引擎 | `AsyncLLM` |

---

## 核心调度（Core Scheduling）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/core/sched/scheduler.py` | 调度器 | `Scheduler`, `Scheduler.schedule()` |
| `vllm/v1/core/sched/request_queue.py` | 请求队列 | `FCFSRequestQueue`, `PriorityRequestQueue` |
| `vllm/v1/core/kv_cache_manager.py` | KV Cache 管理 | `KVCacheManager`, `allocate_slots()`, `free()` |
| `vllm/v1/core/block_pool.py` | 块池管理 | `BlockPool`, `FreeKVCacheBlockQueue` |
| `vllm/v1/core/kv_cache_utils.py` | KV Cache 工具 | `KVCacheBlock`, `BlockHashToBlockMap` |

---

## 请求处理（Request）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/request.py` | 请求数据结构 | `Request`, `RequestStatus` |
| `vllm/sampling_params.py` | 采样参数 | `SamplingParams` |
| `vllm/outputs.py` | 输出数据结构 | `RequestOutput`, `CompletionOutput` |

---

## Worker 执行（Worker）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/worker/gpu_worker.py` | GPU Worker | `GPUWorker` |
| `vllm/v1/worker/gpu_model_runner.py` | 模型执行 | `GPUModelRunner`, `execute_model()` |
| `vllm/v1/worker/gpu_input_batch.py` | 输入批处理 | `InputBatch`, `CachedRequestState` |

---

## Executor 执行器

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/executor/abstract.py` | 执行器基类 | `Executor` |
| `vllm/v1/executor/uniproc_executor.py` | 单进程执行器 | `UniProcExecutor` |
| `vllm/v1/executor/multiproc_executor.py` | 多进程执行器 | `MultiprocExecutor` |
| `vllm/v1/executor/ray_distributed.py` | Ray 分布式执行器 | `RayDistributedExecutor` |

---

## 注意力机制（Attention）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/attention/ops/paged_attn.py` | PagedAttention 接口 | `PagedAttention` |
| `vllm/v1/attention/backends/flash_attn.py` | Flash Attention 后端 | `FlashAttentionBackend` |
| `vllm/v1/attention/backends/triton_attn.py` | Triton Attention 后端 | `TritonAttentionBackend` |
| `vllm/attention/layer.py` | Attention 层 | `Attention` |

---

## 采样（Sampling）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/sample/sampler.py` | 采样器 | `Sampler`, `Sampler.forward()` |
| `vllm/v1/sample/metadata.py` | 采样元数据 | `SamplingMetadata` |
| `vllm/v1/sample/ops/penalties.py` | 惩罚项计算 | `apply_penalties()` |
| `vllm/v1/sample/ops/topk_topp.py` | Top-K/Top-P 采样 | `apply_top_k_top_p()` |

---

## 投机解码（Speculative Decoding）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/v1/spec_decode/eagle.py` | EAGLE 基类 | `SpecDecodeBaseProposer`, `EagleProposer` |
| `vllm/v1/spec_decode/draft_model.py` | Draft Model | `DraftModelProposer` |
| `vllm/v1/spec_decode/medusa.py` | Medusa | `MedusaProposer` |
| `vllm/v1/worker/gpu/spec_decode/rejection_sample.py` | 拒绝采样 | `rejection_sample()` |

---

## 模型实现（Models）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/model_executor/models/llama.py` | LLaMA 模型 | `LlamaForCausalLM` |
| `vllm/model_executor/models/qwen2.py` | Qwen2 模型 | `Qwen2ForCausalLM` |
| `vllm/model_executor/models/mixtral.py` | Mixtral MoE | `MixtralForCausalLM` |
| `vllm/model_executor/models/deepseek_v2.py` | DeepSeek V2 | `DeepseekV2ForCausalLM` |
| `vllm/model_executor/model_loader/loader.py` | 模型加载 | `get_model()` |

---

## 量化（Quantization）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/model_executor/layers/quantization/__init__.py` | 量化入口 | `get_quantization_config()` |
| `vllm/model_executor/layers/quantization/base_config.py` | 量化基类 | `QuantizationConfig` |
| `vllm/model_executor/layers/quantization/fp8.py` | FP8 量化 | `Fp8Config` |
| `vllm/model_executor/layers/quantization/awq.py` | AWQ 量化 | `AWQConfig` |
| `vllm/model_executor/layers/quantization/gptq.py` | GPTQ 量化 | `GPTQConfig` |

---

## 分布式通信（Distributed）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/distributed/parallel_state.py` | 并行状态管理 | `GroupCoordinator` |
| `vllm/distributed/communication_op.py` | 通信操作 | `tensor_model_parallel_all_reduce()` |
| `vllm/distributed/device_communicators/pynccl.py` | NCCL 通信 | `PyNcclCommunicator` |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | 自定义 AllReduce | `CustomAllReduce` |

---

## 配置（Config）

| 文件路径 | 说明 | 关键类/函数 |
|---------|------|------------|
| `vllm/config/vllm.py` | 总配置 | `VllmConfig` |
| `vllm/config/model.py` | 模型配置 | `ModelConfig` |
| `vllm/config/parallel.py` | 并行配置 | `ParallelConfig` |
| `vllm/config/scheduler.py` | 调度器配置 | `SchedulerConfig` |
| `vllm/config/cache.py` | 缓存配置 | `CacheConfig` |

---

## CUDA 内核（CUDA Kernels）

| 文件路径 | 说明 |
|---------|------|
| `csrc/attention/paged_attention_v1.cu` | PagedAttention V1 内核 |
| `csrc/attention/paged_attention_v2.cu` | PagedAttention V2 内核 |
| `csrc/quantization/` | 量化相关内核 |
| `csrc/moe/` | MoE 相关内核 |

---

## 关键函数速查

### 请求处理流程

```python
# 1. 用户调用
LLM.generate()                          # vllm/entrypoints/llm.py

# 2. 引擎处理
LLMEngine.add_request()                  # vllm/v1/engine/llm_engine.py
EngineCore.step()                        # vllm/v1/engine/core.py

# 3. 调度
Scheduler.schedule()                     # vllm/v1/core/sched/scheduler.py
KVCacheManager.allocate_slots()          # vllm/v1/core/kv_cache_manager.py

# 4. 执行
GPUModelRunner.execute_model()           # vllm/v1/worker/gpu_model_runner.py
model.forward()                          # vllm/model_executor/models/*.py

# 5. 采样
Sampler.forward()                        # vllm/v1/sample/sampler.py

# 6. 输出
OutputProcessor.process()                # vllm/v1/engine/processor.py
```

### KV Cache 管理流程

```python
# 分配
KVCacheManager.allocate_slots()          # vllm/v1/core/kv_cache_manager.py
BlockPool.get_free_blocks()              # vllm/v1/core/block_pool.py

# 释放
KVCacheManager.free()                    # vllm/v1/core/kv_cache_manager.py
BlockPool.free_blocks()                  # vllm/v1/core/block_pool.py

# 前缀缓存
KVCacheManager.get_computed_blocks()     # vllm/v1/core/kv_cache_manager.py
```

---

## 调试建议

### 关键断点位置

| 功能 | 文件:行号 | 说明 |
|------|----------|------|
| 请求添加 | `v1/engine/llm_engine.py:add_request` | 追踪请求入口 |
| 调度决策 | `v1/core/sched/scheduler.py:schedule` | 理解调度逻辑 |
| KV 分配 | `v1/core/kv_cache_manager.py:allocate_slots` | 内存分配 |
| 模型执行 | `v1/worker/gpu_model_runner.py:execute_model` | 前向传播 |
| 采样 | `v1/sample/sampler.py:forward` | Token 采样 |

### 日志配置

```bash
# 详细日志
export VLLM_LOGGING_LEVEL=DEBUG

# 函数追踪
export VLLM_TRACE_FUNCTION=1

# 调度器日志
export VLLM_LOG_SCHEDULER=1
```

---

**导航**
- 上一篇：[术语表](glossary.md)
- 下一篇：[参考资料](references.md)
