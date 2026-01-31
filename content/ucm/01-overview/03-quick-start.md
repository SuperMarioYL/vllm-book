---
title: "快速开始"
weight: 3
---


> **阅读时间**: 约 10 分钟
> **前置要求**: [架构概览](./02-architecture-overview.md)

---

## 概述

本文提供 UCM 的快速上手指南，帮助你在最短时间内运行第一个 UCM 增强的推理示例。

---

## 1. 环境准备

### 1.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| Python | 3.10+ | 3.10 |
| PyTorch | 2.0+ | 2.1+ |
| CUDA | 11.8+ | 12.1+ |
| GPU | 16GB+ 显存 | A100 40GB+ |

### 1.2 安装 UCM

```bash
# 方式 1: pip 安装
pip install ucm

git clone https://github.com/your-org/unified-cache-management.git
cd unified-cache-management
# 设置平台
export PLATFORM=cuda  # 或 ascend, musa, maca

pip install -e .
```
### 1.3 验证安装
```bash
python -c "import ucm; print(ucm.__version__)"
```
---
## 2. 基础配置

### 2.1 创建配置文件

创建 `ucm_config.yaml`:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      # 使用 Cache + POSIX 组合
      store_pipeline: "Cache|Posix"
      # KV Cache 存储路径
      storage_backends: "/tmp/ucm_cache"
      # Pinned Memory 缓冲区数量
      buffer_number: 1024
      # 启用 Direct I/O
      io_direct: false

# ucm_sparse_config:
#     sparse_ratio: 0.3
```

### 2.2 创建存储目录

```bash
mkdir -p /tmp/ucm_cache
```

---
## 3. 第一个推理示例
### 3.1 离线推理代码
创建 `inference.py`:
```python
from dataclasses import asdict
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
# 1. 配置 UCM 连接器
ktc = KVTransferConfig(
    kv_connector="UCMConnector",
    kv_connector_module_path="ucm.integration.vllm.ucm_connector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "UCM_CONFIG_FILE": "./ucm_config.yaml"
    }
)
# 2. 配置 vLLM 引擎参数
llm_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",  # 替换为你的模型路径
    kv_transfer_config=ktc,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    enforce_eager=True,  # 建议首次运行使用
)

llm = LLM(**asdict(llm_args))

prompts = [
    "请帮我翻译以下内容：Hello, how are you?",
    "请帮我翻译以下内容：Good morning!",
    "请帮我翻译以下内容：Thank you very much.",
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
)
# 6. 执行推理
print("Starting inference...")
outputs = llm.generate(prompts, sampling_params)
# 7. 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
```
### 3.2 运行推理
```bash
python inference.py
```
### 3.3 预期输出
```
Starting inference...

Prompt: 请帮我翻译以下内容：Hello, how are you?
Generated: 你好，你好吗？
Prompt: 请帮我翻译以下内容：Good morning!
Generated: 早上好！

Prompt: 请帮我翻译以下内容：Thank you very much.
Generated: 非常感谢。
```

---

## 4. 验证 UCM 生效

### 4.1 查看日志

设置详细日志查看 UCM 运行情况：

```bash
export UNIFIED_CACHE_LOG_LEVEL=INFO
python inference.py
```
你应该能看到类似以下日志：
```
[UCM] Initializing UCMDirectConnector...
[UCM] Store pipeline: Cache|Posix
[UCM] Storage backends: /tmp/ucm_cache
[UCM] Request 1: lookup 0 blocks, hit 0
[UCM] Request 1: loading 32 blocks
[UCM] Request 1: saving 32 blocks
[UCM] Request 2: lookup 32 blocks, hit 28  # 第二个请求复用了前缀
```
### 4.2 检查缓存文件
```bash
ls -la /tmp/ucm_cache/

# /tmp/ucm_cache/
# ├── 0/
# │   │   ├── abc123.bin
# │   │   └── def456.bin
# │       └── ...
```
---
## 5. 多轮对话示例

### 5.1 代码示例

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from dataclasses import asdict

ktc = KVTransferConfig(
    kv_connector="UCMConnector",
    kv_connector_module_path="ucm.integration.vllm.ucm_connector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "UCM_CONFIG_FILE": "./ucm_config.yaml"
    }
)

llm_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config=ktc,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)

llm = LLM(**asdict(llm_args))
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

conversation_history = ""

rounds = [
    "你好，请介绍一下你自己。",
    "你能帮我写代码吗？",
    "请用 Python 写一个快速排序。",
]

for i, user_input in enumerate(rounds):
    # 构建完整对话
    conversation_history += f"用户: {user_input}\n助手: "

    # 推理
    outputs = llm.generate([conversation_history], sampling_params)
    response = outputs[0].outputs[0].text

    # 更新历史
    conversation_history += response + "\n"

    print(f"\n=== Round {i+1} ===")
    print(f"用户: {user_input}")
    print(f"助手: {response}")
```

### 5.2 观察 KV 复用

在多轮对话中，你会发现：
- 第一轮：所有 Block 都是新计算的
- 后续轮次：历史对话的 KV 会被复用，只计算新增部分

```
[UCM] Round 1: hit 0 blocks (新对话)
[UCM] Round 2: hit 45 blocks (复用第一轮)
[UCM] Round 3: hit 78 blocks (复用前两轮)
```

---
## 6. 常见问题
### 6.1 ImportError: No module named 'ucm'
```bash
pip install -e .

python -c "import sys; print(sys.path)"
```
### 6.2 CUDA out of memory
```python
llm_args = EngineArgs(
    model="...",
    gpu_memory_utilization=0.6,  # 从 0.8 降低到 0.6
    max_model_len=2048,  # 减少最大长度
)
```
### 6.3 KV Cache 目录权限问题
```bash
chmod -R 755 /tmp/ucm_cache
```
---
## 7. 下一步

完成快速开始后，建议继续阅读：

| 目标 | 推荐文档 |
|------|----------|
| 深入理解存储层 | [存储抽象](../02-storage/01-storage-abstraction.md) |
| 了解稀疏注意力 | [稀疏基础](../03-sparse-attention/01-sparse-fundamentals.md) |
| 生产环境部署 | [部署指南](../06-engineering-practice/01-deployment-guide.md) |
| 性能调优 | [性能调优](../06-engineering-practice/03-performance-tuning.md) |
