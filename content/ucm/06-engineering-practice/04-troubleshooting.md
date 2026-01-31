---
title: "故障排查指南"
weight: 4
---


> **阅读时间**: 约 15 分钟
> **前置要求**: [性能调优](./03-performance-tuning.md)

---

## 概述

本文档提供 UCM 常见问题的诊断和解决方法，包括安装问题、运行时错误、性能问题和集成问题。

---

## 1. 安装问题

### 1.1 编译错误

#### 问题：CUDA 编译失败

**错误信息**：
```
error: CUDA driver version is insufficient for CUDA runtime version
```
**解决方案**：
```bash
# 检查 CUDA 版本
nvidia-smi
nvcc --version
# 确保版本匹配
```

#### 问题：缺少头文件

**错误信息**：
```
fatal error: torch/extension.h: No such file or directory
```
**解决方案**：
```bash
# 确保 PyTorch 安装正确
pip install torch --upgrade

python -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME)"
```
### 1.2 依赖问题
#### 问题：vLLM 版本不兼容
**错误信息**：
```
ImportError: cannot import name 'KVConnectorBase' from 'vllm'
```

**解决方案**：
```bash
pip install vllm==0.9.2

python -c "import vllm; print(vllm.__version__)"
```
#### 问题：Triton 版本问题
**错误信息**：
```
ModuleNotFoundError: No module named 'triton.language'
```

**解决方案**：
```bash
pip install triton==2.0.0
```
---
## 2. 运行时错误

### 2.1 存储错误

#### 问题：存储路径不存在

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: '/data/ucm_cache'
```

**解决方案**：
```bash
mkdir -p /data/ucm_cache

chmod 755 /data/ucm_cache
```

#### 问题：磁盘空间不足

**错误信息**：
```
OSError: [Errno 28] No space left on device
```

**解决方案**：
```bash
df -h /data

rm -rf /data/ucm_cache/*

# ucm_config.yaml
ucm_connector_config:
  auto_cleanup: true
  max_cache_size: 100G
```

### 2.2 内存错误

#### 问题：Pinned Memory 耗尽

**错误信息**：
```
RuntimeError: CUDA out of memory while allocating pinned memory
```

**解决方案**：
```yaml
ucm_connector_config:
  buffer_number: 1024  # 减少缓冲区数量
  pinned_pool_size: 536870912  # 减少到 512 MB
```

#### 问题：GPU 内存不足

**错误信息**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**：
```yaml
ucm_sparse_method: "GSA"
ucm_sparse_config:
  GSA:
    sparse_ratio: 0.2  # 更激进的稀疏比例
```

### 2.3 连接错误

#### 问题：NFS 挂载失败

**错误信息**：
```
mount.nfs: Connection timed out
```

**解决方案**：
```bash
ping nfs-server.local

showmount -e nfs-server.local

sudo iptables -L -n | grep 2049
```

#### 问题：S3 连接失败

**错误信息**：
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**解决方案**：
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

ucm_connector_config:
  s3_access_key: "${AWS_ACCESS_KEY_ID}"
  s3_secret_key: "${AWS_SECRET_ACCESS_KEY}"
```

---
## 3. 集成问题
### 3.1 补丁问题
#### 问题：补丁未生效
**症状**：
- UCM 功能不工作
- 日志中没有 UCM 相关输出
**诊断**：
```python
# 检查补丁状态
import sys
print("'ucm' in modules:", 'ucm' in sys.modules)
print("'vllm' in modules:", 'vllm' in sys.modules)

from vllm.v1.core.sched.output import SchedulerOutput
print("UCM meta:", hasattr(SchedulerOutput, 'ucm_connector_meta'))
```

**解决方案**：
```python
import ucm  # 必须先导入
import vllm  # 然后导入 vLLM
# 或强制应用补丁
from ucm.integration.vllm.patch.apply_patch import apply_all_patches
apply_all_patches()
```

#### 问题：Connector 未被识别

**错误信息**：
```
ValueError: Unknown kv_connector: UCMConnector
```
**解决方案**：
```python
# 检查模块路径
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="UCMConnector",
    # 确保路径正确
    kv_connector_module_path="ucm.integration.vllm.ucm_connector",
    kv_role="kv_both",
)
```

### 3.2 命中率问题

#### 问题：命中率始终为 0

**症状**：
- lookup 总是返回空
- 所有请求都完整计算

**诊断**：
```python
from ucm.store.factory import UcmConnectorFactory

store = UcmConnectorFactory.create_connector(config, 0)

test_ids = [b"test_block_1"]
results = store.lookup(test_ids)
print(f"Lookup results: {results}")

import torch
tensor = torch.randn(1000)
task = store.dump(test_ids, 0, tensor)
store.wait(task)
store.commit(test_ids, [True])

results = store.lookup(test_ids)
print(f"After dump: {results}")
```

**解决方案**：
1. 检查存储路径是否正确
2. 确认 Block 哈希生成一致
3. 检查 commit 是否调用
---
## 4. 性能问题

### 4.1 延迟问题

#### 问题：首次请求延迟高

**原因**：初始化开销

**解决方案**：
```python
llm = LLM(model="...", kv_transfer_config=ktc)

warmup_prompt = "Hello, how are you?"
_ = llm.generate([warmup_prompt])

```

#### 问题：传输延迟高

**诊断**：
```python
from ucm.shared.metrics.ucmmonitor import StatsMonitor

monitor = StatsMonitor.get_instance()
stats = monitor.get_stats()

print(f"Load speed: {stats.get('load_speed', 0) / 1e9:.2f} GB/s")
print(f"Save speed: {stats.get('save_speed', 0) / 1e9:.2f} GB/s")
```

**解决方案**：
```yaml
ucm_connector_config:
  transport_streams: 4

buffer_number: 2048
```

### 4.2 吞吐量问题

#### 问题：批处理吞吐量低

**诊断**：
```python
from ucm.shared.trans.pinned_pool import PinnedMemoryPool

pool = PinnedMemoryPool.get_instance()
stats = pool.get_stats()

usage = stats['currently_used'] / stats['total_allocated']
print(f"Buffer usage: {usage * 100:.1f}%")
```

**解决方案**：
```yaml
ucm_connector_config:
  buffer_number: 4096

ucm_sparse_config:
  GSA:
    prefetch_workers: 8
    prefetch_ahead: 4
```

---
## 5. 调试工具
### 5.1 日志调试
```bash
export UNIFIED_CACHE_LOG_LEVEL=DEBUG
export UCM_PATCH_VERBOSE=1
# 运行
python your_script.py
```
### 5.2 监控调试
```python
from ucm.shared.metrics.ucmmonitor import StatsMonitor

import time

monitor = StatsMonitor.get_instance()

while True:
    stats = monitor.get_stats()
    print(f"\rHit: {stats.get('hit_rate', 0)*100:.1f}% | "
          f"Load: {stats.get('load_speed', 0)/1e9:.1f} GB/s", end="")
    time.sleep(1)
```
### 5.3 性能分析
```python
import torch

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True
) as prof:
    # 运行推理
    output = llm.generate(prompts)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```
---
## 6. 常见问题 FAQ

### Q1: UCM 支持哪些 vLLM 版本？

**A**: 目前完全支持 vLLM v0.9.2，部分支持 v0.9.1。

### Q2: 如何禁用 UCM 补丁？

**A**: 设置环境变量：
```bash
export UCM_DISABLE_PATCHES=1
```

### Q3: 如何清理缓存？

**A**:
```bash
rm -rf /data/ucm_cache/*

from ucm.store.factory import UcmConnectorFactory
store = UcmConnectorFactory.create_connector(config, 0)
store.clear()
```

### Q4: UCM 是否支持 TP/PP？

**A**: UCM 支持张量并行（TP），每个 rank 有独立的 Connector。

### Q5: 如何查看当前配置？

**A**:
```python
from ucm.config import get_current_config
config = get_current_config()
print(config)
```

---
## 7. 获取帮助
### 7.1 问题报告
在报告问题时，请提供：
1. **环境信息**：
   ```bash
   python -c "import ucm; print(ucm.__version__)"
   python -c "import vllm; print(vllm.__version__)"
   python -c "import torch; print(torch.__version__)"
   nvidia-smi
   ```
2. **配置文件**：UCM 配置和 vLLM 配置
3. **错误日志**：完整的错误堆栈
4. **复现步骤**：最小化的复现代码
### 7.2 社区资源
- GitHub Issues: https://github.com/anthropics/claude-code/issues
- 文档: 本文档系列

---

## 8. 故障排查清单

### 安装检查
- [ ] CUDA 驱动和工具链版本匹配
- [ ] PyTorch 安装正确
- [ ] vLLM 版本为 v0.9.2
- [ ] UCM 编译成功

### 配置检查
- [ ] 存储路径存在且有写权限
- [ ] buffer_number >= 1024
- [ ] block_size 与 vLLM 一致
### 运行时检查
- [ ] UCM 在 vLLM 之前导入
- [ ] 补丁正确应用
- [ ] Connector 正确创建
- [ ] 存储连接正常
### 性能检查
- [ ] 命中率 > 50%
- [ ] 传输速度 > 1 GB/s
- [ ] 无内存泄漏
