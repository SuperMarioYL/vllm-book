---
title: "代码索引"
weight: 1
---

本文档提供 CacheBlend 核心实现代码的快速索引，帮助开发者快速定位关键功能的实现位置。

---

## 代码位置索引

| 功能 | 文件 | 位置 |
|------|------|------|
| cache_fuse_metadata 定义 | llama.py | 第 300-310 行 |
| old_kvs 定义 | llama.py | 第 312 行 |
| 状态机实现 | llama.py | 第 330-376 行 |
| RoPE 位置恢复 | llama.py | 第 174-179 行 |
| hack_kv 收集 | llama.py | 第 181-182 行 |
| HKVD 选择 | xformers.py | 第 204-221 行 |
| KV 融合 | xformers.py | 第 240-245 行 |
| 注意力计算 | xformers.py | 第 426-444 行 |

---

## 核心组件说明

### cache_fuse_metadata

存储 CacheBlend 运行时所需的元数据，包括：
- `check`: 是否启用 CacheBlend
- `collect`: 是否收集 KV
- `check_layers`: HKVD 选择层
- `recomp_ratio`: 重计算比例
- `suffix_len`: 后缀长度

### old_kvs

存储预计算的 KV Cache，用于后续的融合操作。

### 状态机实现

控制 CacheBlend 的三种状态：
- 状态 0: 普通 Prefill
- 状态 1: HKVD 选择阶段
- 状态 2: 选择性重计算阶段

---

## 参考文献

1. CacheBlend 论文: https://arxiv.org/abs/2405.16444
2. LMCache 项目: https://github.com/LMCache/LMCache
