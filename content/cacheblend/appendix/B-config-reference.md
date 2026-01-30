---
title: "配置参考"
weight: 2
---

本文档详细说明 CacheBlend 的所有配置参数及其使用方法。

---

## 核心配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `check` | False | 是否启用 CacheBlend |
| `collect` | False | 是否收集 KV |
| `check_layers` | [1] | HKVD 选择层 |
| `recomp_ratio` | 0.16 | 重计算比例 |
| `suffix_len` | - | 后缀长度（必须设置） |

---

## 参数详解

### check

**类型**: `bool`

**说明**: 控制是否启用 CacheBlend 的选择性重计算功能。当设置为 `True` 时，系统将使用预计算的 KV Cache 并进行部分重计算。

**示例**:
```python
cache_fuse_metadata['check'] = True
```

### collect

**类型**: `bool`

**说明**: 控制是否收集当前推理的 KV Cache。用于预计算文本块的 KV Cache 供后续复用。

**示例**:
```python
cache_fuse_metadata['collect'] = True
llm.generate([context], SamplingParams(max_tokens=1))
cache_fuse_metadata['collect'] = False
```

### check_layers

**类型**: `List[int]`

**说明**: 指定进行 HKVD token 选择的层。默认为 Layer 1，这是论文中验证的最佳位置。

**示例**:
```python
cache_fuse_metadata['check_layers'] = [1]  # 单层选择
cache_fuse_metadata['check_layers'] = [1, 8, 16]  # 多层选择（实验性）
```

### recomp_ratio

**类型**: `float`

**说明**: 指定需要重计算的 token 比例。取值范围为 0.0 到 1.0。

**推荐值**:
- 高质量要求: 0.20-0.30
- 平衡模式: 0.15-0.20
- 高速度要求: 0.08-0.15

**示例**:
```python
cache_fuse_metadata['recomp_ratio'] = 0.16
```

### suffix_len

**类型**: `int`

**说明**: 指定输入中后缀（查询部分）的 token 长度。这个参数必须正确设置，否则会影响 HKVD 选择的准确性。

**示例**:
```python
suffix_len = len(tokenizer.encode(query))
cache_fuse_metadata['suffix_len'] = suffix_len
```

---

## 配置示例

### 基础配置

```python
# 获取 metadata 引用
cache_fuse_metadata = llm.llm_engine.model_executor.\
    driver_worker.model_runner.model.model.cache_fuse_metadata

# 收集阶段
cache_fuse_metadata['collect'] = True
cache_fuse_metadata['check'] = False
llm.generate([context], SamplingParams(max_tokens=1))

# 融合阶段
cache_fuse_metadata['collect'] = False
cache_fuse_metadata['check'] = True
cache_fuse_metadata['suffix_len'] = len(tokenizer.encode(query))
cache_fuse_metadata['recomp_ratio'] = 0.16
output = llm.generate([context + query], sampling_params)
```

### 高质量配置

```python
cache_fuse_metadata['check_layers'] = [0]  # 使用更早的层
cache_fuse_metadata['recomp_ratio'] = 0.25  # 更高的重计算比例
```

### 高速度配置

```python
cache_fuse_metadata['check_layers'] = [2]  # 使用稍晚的层
cache_fuse_metadata['recomp_ratio'] = 0.10  # 更低的重计算比例
```

---

## 参考文献

1. CacheBlend 论文: https://arxiv.org/abs/2405.16444
