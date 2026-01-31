---
title: "基础设施调试指南"
weight: 4
---


> **阅读时间**: 约 10 分钟
> **前置要求**: [日志系统](./03-logging-system.md)

---

## 概述

本文提供 UCM 共享基础设施的调试入口点、验证方法和问题排查技巧。

---

## 1. 调试入口点

### 1.1 入口点清单

```
基础设施调试入口
================

1. 传输层
   ├── ucm/shared/trans/cuda_transport.py:50
   │   CUDATransport.__init__()
   ├── ucm/shared/trans/cuda_transport.py:80
   │   CUDATransport.h2d()
   └── ucm/shared/trans/pinned_pool.py:30
       PinnedMemoryPool.__init__()

2. 监控系统
   ├── ucm/shared/metrics/observability.py:46
   │   PrometheusLogger.__init__()
   ├── ucm/shared/metrics/observability.py:100
   │   PrometheusLogger.log_lookup()
   └── ucm/shared/metrics/ucmmonitor.py:20
       StatsMonitor.get_instance()

3. 日志系统
   └── ucm/logger.py:29
       init_logger()
```
### 1.2 调试流程
```mermaid
flowchart TB
    subgraph debug_flow["调试流程"]
        START["开始调试"]
        CHECK_LOG["1. 检查日志配置"]
        CHECK_TRANS["2. 验证传输层"]
        CHECK_METRICS["3. 检查监控指标"]
        CHECK_PERF["4. 性能分析"]
        END["完成"]
        START --> CHECK_LOG
        CHECK_LOG --> CHECK_TRANS
        CHECK_TRANS --> CHECK_METRICS
        CHECK_METRICS --> CHECK_PERF
        CHECK_PERF --> END
    end
```
---
## 2. 传输层调试

### 2.1 验证传输功能

```python
#!/usr/bin/env python
"""验证传输层功能"""

import torch
from ucm.shared.trans import CUDATransport


def verify_transport():
    print("=== Transport Layer Verification ===\n")

    # 1. 检查 CUDA 可用性
    print(f"1. CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("   CUDA not available, skipping GPU tests")
        return

    # 2. 创建传输实例
    try:
        transport = CUDATransport(device_id=0)
        print("2. CUDATransport created: OK")
    except Exception as e:
        print(f"2. CUDATransport creation failed: {e}")
        return

    # 3. 测试 Pinned Memory
    try:
        pinned = transport.get_pinned_buffer(1024 * 1024)  # 1 MB
        print(f"3. Pinned buffer allocated: {pinned.shape}, {pinned.is_pinned()}")
        transport.release_pinned_buffer(pinned)
        print("   Pinned buffer released: OK")
    except Exception as e:
        print(f"3. Pinned memory test failed: {e}")

    # 4. 测试 H2D 传输
    try:
        cpu_tensor = torch.randn(1024, 1024)
        gpu_tensor = torch.empty_like(cpu_tensor, device='cuda:0')

        # 使用 Pinned Memory
        pinned = transport.get_pinned_buffer(cpu_tensor.numel() * 4)
        pinned_view = pinned.view(cpu_tensor.shape).float()
        pinned_view.copy_(cpu_tensor)

        task = transport.h2d(pinned_view, gpu_tensor)
        transport.synchronize(task)

        # 验证数据
        diff = (cpu_tensor - gpu_tensor.cpu()).abs().max()
        print(f"4. H2D transfer: OK (max diff: {diff})")

        transport.release_pinned_buffer(pinned)
    except Exception as e:
        print(f"4. H2D transfer failed: {e}")

    # 5. 测试 D2H 传输
    try:
        gpu_tensor = torch.randn(1024, 1024, device='cuda:0')
        pinned = transport.get_pinned_buffer(gpu_tensor.numel() * 4)
        cpu_tensor = pinned.view(gpu_tensor.shape).float()

        task = transport.d2h(gpu_tensor, cpu_tensor)
        transport.synchronize(task)

        diff = (gpu_tensor.cpu() - cpu_tensor).abs().max()
        print(f"5. D2H transfer: OK (max diff: {diff})")

        transport.release_pinned_buffer(pinned)
    except Exception as e:
        print(f"5. D2H transfer failed: {e}")

    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    verify_transport()
```

### 2.2 Pinned Memory 诊断

```python
def diagnose_pinned_memory():
    """诊断 Pinned Memory 池"""
    from ucm.shared.trans.pinned_pool import PinnedMemoryPool

    pool = PinnedMemoryPool(device_id=0)

    # 获取统计
    stats = pool.get_stats()
    print("Pinned Memory Pool Stats:")
    print(f"  Total allocated: {stats['total_allocated'] / 1e9:.2f} GB")
    print(f"  Currently used: {stats['currently_used']}")
    print(f"  Total free: {stats['total_free'] / 1e6:.1f} MB")
    print(f"  Peak allocated: {stats['peak_allocated']}")

    # 检查碎片化
    fragmentation = 1 - (stats['total_free'] / stats['total_allocated'])
    print(f"  Fragmentation: {fragmentation * 100:.1f}%")
```

---
## 3. 监控系统调试
### 3.1 验证 Prometheus 指标
```python
def verify_prometheus_metrics():
    """验证 Prometheus 指标收集"""
    from ucm.shared.metrics.observability import PrometheusLogger

    logger = PrometheusLogger(labels={'service': 'ucm_test'})

    # 记录一些测试数据
    logger.log_lookup(total=100, hits=85)
    logger.log_load(blocks=10, duration=0.5, bytes_loaded=1024000)
    logger.log_save(blocks=5, duration=0.3, bytes_saved=512000)
    print("Prometheus metrics recorded")
    # 如果启用了 HTTP 端点
    try:
        import requests
        response = requests.get('http://localhost:9090/metrics')
        print(f"Metrics endpoint: {response.status_code}")
        # 检查关键指标
        metrics = response.text
        expected = ['ucm_lookup_total', 'ucm_load_blocks_total', 'ucm_save_blocks_total']
        for metric in expected:
            if metric in metrics:
                print(f"  {metric}: FOUND")
            else:
                print(f"  {metric}: MISSING")
    except Exception as e:
        print(f"Cannot access metrics endpoint: {e}")
```
### 3.2 StatsMonitor 诊断
```python
def diagnose_stats_monitor():
    """诊断 StatsMonitor"""
    from ucm.shared.metrics.ucmmonitor import StatsMonitor

    monitor = StatsMonitor.get_instance()

    # 模拟一些操作
    for i in range(10):
        monitor.record_lookup(total=100, hits=80 + i)
        monitor.record_load(blocks=10, bytes=1024000, duration=0.1)
    # 获取统计
    stats = monitor.get_stats()

    print("StatsMonitor Stats:")
    print(f"  Lookup count: {stats['lookup_count']}")
    print(f"  Hit rate: {stats.get('hit_rate', 0) * 100:.1f}%")
    print(f"  Load speed: {stats.get('load_speed', 0) / 1e6:.1f} MB/s")
```

---

## 4. 日志系统调试

### 4.1 验证日志配置

```python
def verify_logging():
    """验证日志配置"""
    import logging
    import os

    # 检查环境变量
    log_level = os.environ.get('UNIFIED_CACHE_LOG_LEVEL', 'INFO')
    print(f"Log level from env: {log_level}")

    # 检查日志器配置
    from ucm.logger import get_logger
    logger = get_logger('test')
    print(f"Logger level: {logging.getLevelName(logger.level)}")
    print(f"Logger handlers: {len(logger.handlers)}")
    for handler in logger.handlers:
        print(f"  Handler: {type(handler).__name__}")
        print(f"    Level: {logging.getLevelName(handler.level)}")
        print(f"    Formatter: {handler.formatter._fmt if handler.formatter else 'None'}")

    # 测试各级别日志
    print("\nTest log output:")
    logger.debug("DEBUG message")
    logger.info("INFO message")
    logger.warning("WARNING message")
    logger.error("ERROR message")
```

### 4.2 日志级别调整

```bash
# 临时开启 DEBUG
export UNIFIED_CACHE_LOG_LEVEL=DEBUG

python -c "
from ucm.logger import get_logger
logger = get_logger('debug_test')
logger.debug('This should appear')
"

export UNIFIED_CACHE_LOG_LEVEL=INFO
```
---
## 5. 常见问题排查

### 5.1 传输层问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| CUDA 不可用 | 驱动/环境问题 | 检查 `nvidia-smi`，重装驱动 |
| Pinned Memory 耗尽 | 池大小不足 | 增加 `pinned_pool_size` |
| 传输超时 | PCIe 带宽受限 | 检查带宽，减少并发 |
| 数据不一致 | 同步问题 | 确保正确调用 `synchronize` |

### 5.2 监控问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 指标不更新 | Logger 未初始化 | 检查 `PrometheusLogger` 创建 |
| 端点不可访问 | 端口被占用 | 更换端口或检查防火墙 |
| 数据丢失 | 采集间隔太长 | 调整 `collection_interval` |

### 5.3 日志问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 日志不输出 | 级别设置过高 | 降低日志级别 |
| 重复日志 | Handler 重复添加 | 检查 `propagate` 设置 |
| 性能下降 | DEBUG 级别开启 | 生产环境使用 INFO |

---
## 6. 性能分析
### 6.1 传输层性能
```python
import time
import torch
from ucm.shared.trans import CUDATransport

def benchmark_transport():
    """传输层性能基准测试"""
    transport = CUDATransport(device_id=0)
    sizes = [1, 4, 16, 64, 256]  # MB
    print("Transfer Benchmark:")
    print("-" * 50)

    for size_mb in sizes:
        size = size_mb * 1024 * 1024
        iterations = 10

        # 准备数据
        pinned = transport.get_pinned_buffer(size)
        gpu = torch.empty(size, dtype=torch.uint8, device='cuda:0')

        # H2D 测试
        start = time.time()
        for _ in range(iterations):
            task = transport.h2d(pinned, gpu)
            transport.synchronize(task)
        h2d_time = (time.time() - start) / iterations
        h2d_bw = size / h2d_time / 1e9

        # D2H 测试
        start = time.time()
        for _ in range(iterations):
            task = transport.d2h(gpu, pinned)
            transport.synchronize(task)
        d2h_time = (time.time() - start) / iterations
        d2h_bw = size / d2h_time / 1e9

        print(f"{size_mb:3d} MB: H2D {h2d_bw:.1f} GB/s, D2H {d2h_bw:.1f} GB/s")

        transport.release_pinned_buffer(pinned)

if __name__ == "__main__":
    benchmark_transport()
```
### 6.2 预期性能
| 传输类型 | PCIe 3.0 | PCIe 4.0 | NVLink |
|----------|----------|----------|--------|
| H2D | ~12 GB/s | ~24 GB/s | ~100 GB/s |
| D2H | ~12 GB/s | ~24 GB/s | ~100 GB/s |

---

## 7. 调试清单

### 7.1 传输层

- [ ] CUDA 驱动正常
- [ ] Pinned Memory 可分配
- [ ] H2D 传输正常
- [ ] D2H 传输正常
- [ ] 传输带宽符合预期

### 7.2 监控系统

- [ ] PrometheusLogger 初始化
- [ ] 指标正确记录
- [ ] 端点可访问
- [ ] Grafana 连接正常
### 7.3 日志系统
- [ ] 日志级别正确
- [ ] Handler 正确配置
- [ ] 日志输出正常
- [ ] 无重复日志

---

## 8. 环境变量参考

```bash
# 日志级别
UNIFIED_CACHE_LOG_LEVEL=DEBUG

UCM_STORE_LOG_LEVEL=INFO
UCM_SPARSE_LOG_LEVEL=DEBUG
UCM_INTEGRATION_LOG_LEVEL=INFO

UCM_PINNED_POOL_SIZE=1073741824  # 1 GB
UCM_TRANSPORT_STREAMS=2
# 监控
UCM_METRICS_ENABLED=true
UCM_METRICS_PORT=9090
```
