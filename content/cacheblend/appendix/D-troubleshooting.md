---
title: "问题排查"
weight: 4
---

本文档提供 CacheBlend 常见问题的诊断和解决方案。

---

## 常见问题与解决方案

### 问题 1: 输出质量明显下降

**症状**: CacheBlend 生成的输出与完整 Prefill 差异很大

**可能原因与解决方案**:

1. **重计算比例太低**
   ```python
   # 调高重计算比例
   cache_fuse_metadata['recomp_ratio'] = 0.25  # 从 0.16 增加到 0.25
   ```

2. **Check 层选择不当**
   ```python
   # 尝试使用更早或更晚的层
   cache_fuse_metadata['check_layers'] = [0]  # 或 [2]
   ```

3. **后缀长度设置错误**
   ```python
   # 确保后缀长度正确计算
   suffix_len = len(query_tokens)  # 而不是 len(query_prompt)
   cache_fuse_metadata['suffix_len'] = suffix_len
   ```

### 问题 2: TTFT 没有明显改善

**症状**: 启用 CacheBlend 后 TTFT 与完整 Prefill 相近

**可能原因与解决方案**:

1. **输入太短**
   - CacheBlend 对短输入的优势不明显
   - 建议输入长度 > 1000 tokens

2. **KV Cache 加载延迟太高**
   ```python
   # 检查 KV Cache 存储位置
   # 优先使用 GPU 内存 > CPU 内存 > SSD
   ```

3. **重计算比例太高**
   ```python
   cache_fuse_metadata['recomp_ratio'] = 0.10  # 降低重计算比例
   ```

### 问题 3: CUDA 内存溢出

**症状**: RuntimeError: CUDA out of memory

**解决方案**:

```python
# 1. 降低 GPU 内存利用率
llm = LLM(model="...", gpu_memory_utilization=0.4)

# 2. 分块处理长输入
# 将输入分成更小的 chunk

# 3. 清理旧的 KV Cache
llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = \
    [[None, None]] * num_layers
```

---

## 性能诊断

### 诊断 TTFT 组成

```python
import time

# 测量各阶段时间
t1 = time.time()
# KV Cache 加载
t2 = time.time()
# HKVD 选择
t3 = time.time()
# 部分重计算
t4 = time.time()
# Token 生成
t5 = time.time()

print(f"KV 加载: {t2-t1:.3f}s")
print(f"HKVD 选择: {t3-t2:.3f}s")
print(f"部分重计算: {t4-t3:.3f}s")
print(f"生成: {t5-t4:.3f}s")
```

### 验证 HKVD 选择质量

```python
# 检查选择的 HKVD indices
imp_indices = cache_fuse_metadata["imp_indices"]
print(f"HKVD token 数量: {len(imp_indices)}")
print(f"HKVD token 比例: {len(imp_indices) / total_len:.2%}")

# 可视化 KV 偏差分布
import matplotlib.pyplot as plt
temp_diff = torch.sum((value - value_old)**2, dim=[1,2])
plt.hist(temp_diff.cpu().numpy(), bins=50)
plt.title("KV Deviation Distribution")
plt.savefig("kv_deviation.png")
```

---

## 日志与监控

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在关键位置添加日志
logger = logging.getLogger(__name__)

# 在 xformers.py 中
if status == 1:
    logger.debug(f"HKVD 选择: {len(top_indices)} tokens from {total_len}")
    logger.debug(f"Top-K 偏差范围: {temp_diff[top_indices].min():.4f} - {temp_diff[top_indices].max():.4f}")
```

---

## 参考文献

1. CacheBlend 论文: https://arxiv.org/abs/2405.16444
