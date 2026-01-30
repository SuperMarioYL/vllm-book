---
title: "RoPE 恢复算法"
weight: 1
---

本文档详细介绍 CacheBlend 中 RoPE（Rotary Position Embedding）位置恢复算法的原理与实现。该算法是多 Chunk KV Cache 拼接的基础，确保预计算的 KV Cache 在新上下文中保持正确的位置编码。

---

## 5.1.1 问题背景

预计算的 KV Cache 中的 K 向量是用原始位置（从 0 开始）旋转的。当多个 chunk 拼接时，需要将 K 向量重新旋转到正确的全局位置。

## 5.1.2 数学原理

根据论文附录 A，RoPE 的关键性质是注意力分数只依赖相对位置：

$$q_{m+l}k_m = \sum_{i=0}^{d/2-1}(q_{[2i]}k_{[2i]} + q_{[2i+1]}k_{[2i+1]})\cos l\theta_i$$

其中 $l = (m+l) - m$ 是相对距离。

因此，只需要确保 K 向量的位置编码与实际拼接后的位置一致即可。

## 5.1.3 代码实现

```python
# 文件: vllm_blend/vllm/model_executor/models/llama.py
# 位置: LlamaAttention.forward() 第 174-179 行

if status in [1, 2]:
    # 创建虚拟 Query（只需要形状匹配）
    if cache_fuse_metadata["fake_q"] is None:
        cache_fuse_metadata['fake_q'] = torch.rand_like(q)

    # 使用原始位置重新旋转旧 K
    _, old_kv[0] = self.rotary_emb(
        cache_fuse_metadata['org_pos'],  # 原始位置 [0, 1, 2, ..., seq_len-1]
        cache_fuse_metadata['fake_q'],   # 虚拟 Q（不使用其输出）
        old_kv[0]                        # 旧 K
    )
```

## 5.1.4 算法流程

```mermaid
graph TB
    subgraph rope["RoPE 位置恢复"]
        OldK["旧 K (原始位置编码)"] --> Check{status in [1,2]?}
        Check -->|是| CreateFakeQ["创建 fake_q"]
        CreateFakeQ --> RotaryEmb["rotary_emb(org_pos, fake_q, old_K)"]
        RotaryEmb --> NewOldK["旧 K (正确位置编码)"]
        Check -->|否| Skip["跳过"]
    end
```

## 5.1.5 正确性证明

由于 RoPE 只依赖相对位置，旋转操作是可逆的：

1. **原始旋转**：$K_{orig} = R_{\Theta, pos_{orig}} \cdot K$
2. **反向旋转**：$K = R_{\Theta, -pos_{orig}} \cdot K_{orig}$
3. **正确旋转**：$K_{correct} = R_{\Theta, pos_{correct}} \cdot K$

由于 `rotary_emb` 函数直接应用给定位置的旋转，调用它会直接得到正确位置的 K。

---

## 下一步

- [HKVD Token 选择算法](./02-hkvd-selection.md) - 了解如何选择需要重计算的关键 Token
