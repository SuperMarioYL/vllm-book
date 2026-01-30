---
title: "基准测试"
weight: 5
---

本文档提供 CacheBlend 性能基准测试的完整代码，包括 TTFT 测试和质量测试。

---

## TTFT 基准测试

```python
import time
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def benchmark_ttft(model_name, input_lengths, num_trials=5):
    """测试不同输入长度下的 TTFT"""

    llm = LLM(model=model_name, gpu_memory_utilization=0.5)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm.set_tokenizer(tokenizer)

    results = {}

    for length in input_lengths:
        # 生成测试输入
        test_input = "Hello " * (length // 2)

        ttfts = {
            'full_prefill': [],
            'cacheblend': []
        }

        for trial in range(num_trials):
            # 测试完整 Prefill
            cache_fuse_metadata = llm.llm_engine.model_executor.\
                driver_worker.model_runner.model.model.cache_fuse_metadata
            cache_fuse_metadata['check'] = False
            cache_fuse_metadata['collect'] = False

            sampling_params = SamplingParams(temperature=0, max_tokens=10)
            output = llm.generate([test_input], sampling_params)
            ttft_full = output[0].metrics.first_token_time - \
                       output[0].metrics.first_scheduled_time
            ttfts['full_prefill'].append(ttft_full)

            # 测试 CacheBlend
            # 首先收集 KV
            cache_fuse_metadata['collect'] = True
            llm.generate([test_input[:length//2]],
                        SamplingParams(temperature=0, max_tokens=1))

            # 然后使用 CacheBlend
            cache_fuse_metadata['check'] = True
            cache_fuse_metadata['collect'] = False
            cache_fuse_metadata['suffix_len'] = length // 4

            output = llm.generate([test_input], sampling_params)
            ttft_blend = output[0].metrics.first_token_time - \
                        output[0].metrics.first_scheduled_time
            ttfts['cacheblend'].append(ttft_blend)

        results[length] = {
            'full_prefill_mean': sum(ttfts['full_prefill']) / num_trials,
            'full_prefill_std': torch.std(torch.tensor(ttfts['full_prefill'])).item(),
            'cacheblend_mean': sum(ttfts['cacheblend']) / num_trials,
            'cacheblend_std': torch.std(torch.tensor(ttfts['cacheblend'])).item(),
        }

        print(f"Length {length}:")
        print(f"  Full Prefill: {results[length]['full_prefill_mean']:.3f}s ± {results[length]['full_prefill_std']:.3f}s")
        print(f"  CacheBlend:   {results[length]['cacheblend_mean']:.3f}s ± {results[length]['cacheblend_std']:.3f}s")
        print(f"  Speedup:      {results[length]['full_prefill_mean'] / results[length]['cacheblend_mean']:.2f}x")

    return results

# 运行基准测试
if __name__ == "__main__":
    results = benchmark_ttft(
        "mistralai/Mistral-7B-Instruct-v0.2",
        [512, 1024, 2048, 4096, 8192]
    )
```

---

## 质量基准测试

```python
from datasets import load_dataset
from rouge_score import rouge_scorer

def benchmark_quality(model_name, dataset_name, num_samples=100):
    """在标准数据集上测试生成质量"""

    llm = LLM(model=model_name, gpu_memory_utilization=0.5)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm.set_tokenizer(tokenizer)

    # 加载数据集
    if dataset_name == "2wikimqa":
        dataset = load_dataset("THUDM/2WikiMultihopQA", split="validation")
        metric = "f1"
    elif dataset_name == "samsum":
        dataset = load_dataset("samsum", split="test")
        metric = "rouge-l"

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    results = {
        'full_prefill': [],
        'cacheblend': [],
        'full_kv_reuse': []
    }

    cache_fuse_metadata = llm.llm_engine.model_executor.\
        driver_worker.model_runner.model.model.cache_fuse_metadata

    for i, sample in enumerate(dataset.select(range(num_samples))):
        context = sample['context'] if 'context' in sample else sample['dialogue']
        question = sample['question'] if 'question' in sample else "Summarize:"
        reference = sample['answer'] if 'answer' in sample else sample['summary']

        prompt = f"{context}\n\n{question}"
        sampling_params = SamplingParams(temperature=0, max_tokens=100)

        # 方法 1: Full Prefill
        cache_fuse_metadata['check'] = False
        cache_fuse_metadata['collect'] = False
        output = llm.generate([prompt], sampling_params)
        pred = output[0].outputs[0].text

        if metric == "f1":
            score = compute_f1(pred, reference)
        else:
            score = scorer.score(reference, pred)['rougeL'].fmeasure
        results['full_prefill'].append(score)

        # 方法 2: CacheBlend
        # 首先收集 context 的 KV
        cache_fuse_metadata['collect'] = True
        llm.generate([context], SamplingParams(temperature=0, max_tokens=1))

        # 使用 CacheBlend
        cache_fuse_metadata['check'] = True
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['suffix_len'] = len(tokenizer.encode(question))

        output = llm.generate([prompt], sampling_params)
        pred = output[0].outputs[0].text

        if metric == "f1":
            score = compute_f1(pred, reference)
        else:
            score = scorer.score(reference, pred)['rougeL'].fmeasure
        results['cacheblend'].append(score)

        if i % 10 == 0:
            print(f"Processed {i+1}/{num_samples} samples")

    # 计算平均分数
    print("\n=== 质量基准测试结果 ===")
    print(f"Full Prefill:    {sum(results['full_prefill'])/len(results['full_prefill']):.4f}")
    print(f"CacheBlend:      {sum(results['cacheblend'])/len(results['cacheblend']):.4f}")
    print(f"差异:            {abs(sum(results['full_prefill'])/len(results['full_prefill']) - sum(results['cacheblend'])/len(results['cacheblend'])):.4f}")

    return results

def compute_f1(pred, ref):
    """计算 F1 分数"""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
```

---

## 运行说明

### 依赖安装

```bash
pip install vllm transformers datasets rouge-score torch
```

### 运行 TTFT 测试

```bash
python benchmark_ttft.py
```

### 运行质量测试

```bash
python benchmark_quality.py
```

---

## 参考文献

1. CacheBlend 论文: https://arxiv.org/abs/2405.16444
