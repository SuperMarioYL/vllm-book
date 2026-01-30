---
title: "快速开始"
weight: 2
---

## 概述

### 本章学习目标
- 完成 SGLang 的安装
- 启动第一个推理服务
- 使用多种方式发送请求
- 了解基本配置选项

### 前置知识要求
- 熟悉 Python 环境管理
- 了解 HTTP API 基础
- 具备 NVIDIA GPU 环境

---

## 安装 SGLang

### 方法一：pip 安装（推荐）

```bash
# 创建新的 conda 环境
conda create -n sglang python=3.10 -y
conda activate sglang

# 安装 SGLang
pip install "sglang[all]"

# 安装 FlashInfer（推荐的注意力后端）
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### 方法二：从源码安装

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### 方法三：Docker

```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

### 验证安装

```python
import sglang as sgl
print(sgl.__version__)
```

---

## 启动推理服务

### 基本启动命令

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 30000
```

### 常用启动参数

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --mem-fraction-static 0.88 \
    --context-length 8192
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径或 HuggingFace 模型名 | 必填 |
| `--port` | 服务端口 | 30000 |
| `--host` | 监听地址 | 127.0.0.1 |
| `--tensor-parallel-size` | 张量并行 GPU 数 | 1 |
| `--mem-fraction-static` | GPU 内存使用比例 | 0.88 |
| `--context-length` | 最大上下文长度 | 模型默认 |

### 启动成功日志

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
```

---

## 发送请求

### 方法一：OpenAI 兼容 API

SGLang 兼容 OpenAI API 格式：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"  # SGLang 不需要真实 API key
)

# Chat Completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 方法二：原生 API

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "What is the capital of France?",
        "sampling_params": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }
)

print(response.json()["text"])
```

### 方法三：流式输出

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a short story about a robot.",
        "sampling_params": {
            "max_new_tokens": 200,
            "temperature": 0.8
        },
        "stream": True
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode(), end="", flush=True)
```

### 方法四：SGLang Frontend DSL

```python
import sglang as sgl

@sgl.function
def chat(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))

# 连接到服务器
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

# 运行
state = chat.run(question="What is machine learning?")
print(state["answer"])
```

### 方法五：curl

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 50
    }'
```

---

## 离线批量推理

### 使用 sgl.Engine

```python
import sglang as sgl

# 创建离线引擎
engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# 批量推理
prompts = [
    "What is AI?",
    "Explain quantum computing.",
    "What is the meaning of life?"
]

outputs = engine.generate(prompts, sampling_params={"max_new_tokens": 100})

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output['text']}\n")

# 关闭引擎
engine.shutdown()
```

### 使用 generate 函数

```python
import sglang as sgl

@sgl.function
def batch_qa(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=50))

# 批量运行
questions = ["What is 2+2?", "Who is Einstein?", "What is Python?"]
states = batch_qa.run_batch(
    [{"question": q} for q in questions],
    backend=sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
)

for q, state in zip(questions, states):
    print(f"Q: {q}")
    print(f"A: {state['answer']}\n")
```

---

## 配置选项详解

### 内存配置

```bash
# 控制 KV Cache 内存比例
--mem-fraction-static 0.85

# 限制最大运行请求数
--max-running-requests 256

# 限制最大 Prefill Token 数
--max-prefill-tokens 16384
```

### 性能配置

```bash
# 启用 Chunked Prefill
--chunked-prefill-size 8192

# 调度策略
--schedule-policy lpm  # lpm, fcfs, random

# CUDA Graph 批大小
--cuda-graph-max-bs 128
```

### 模型配置

```bash
# 量化
--quantization fp8
--quantization awq

# KV Cache 量化
--kv-cache-dtype fp8_e5m2

# LoRA
--enable-lora
--lora-paths my-lora=/path/to/lora
```

### 分布式配置

```bash
# 张量并行
--tensor-parallel-size 4

# 流水线并行
--pipeline-parallel-size 2

# 数据并行
--data-parallel-size 2
```

---

## 常见问题

### 1. CUDA 内存不足

```bash
# 减少内存使用
--mem-fraction-static 0.8
--max-running-requests 128

# 使用量化
--quantization fp8
```

### 2. 模型加载慢

```bash
# 使用本地缓存
export HF_HOME=/path/to/cache

# 或指定本地路径
--model /local/path/to/model
```

### 3. 端口被占用

```bash
# 更换端口
--port 30001

# 或释放端口
lsof -i :30000
kill -9 <PID>
```

---

## 小结

### 要点回顾

1. **安装方式**：pip、源码、Docker 三种方式
2. **启动服务**：`python -m sglang.launch_server`
3. **发送请求**：OpenAI API、原生 API、DSL、curl
4. **离线推理**：`sgl.Engine` 进行批量处理
5. **配置调优**：内存、性能、模型、分布式配置

### 下一章预告

在下一章《架构总览》中，我们将：
- 了解 SGLang 的三进程架构
- 理解各组件的职责
- 掌握数据流向
