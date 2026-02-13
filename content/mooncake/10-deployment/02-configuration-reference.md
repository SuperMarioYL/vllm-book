---
title: "配置参考手册"
weight: 2
---

[上一篇](01-deployment-guide.md) | [目录](../README.md)

# 配置参考手册

本章提供 Mooncake 各组件的完整配置参考，涵盖 MasterService、Transfer Engine、TENT、客户端以及所有环境变量的详细说明。

## 13.2 MasterService 配置

### 13.2.1 配置文件格式

MasterService 支持 JSON 和 YAML 两种配置文件格式。配置文件通常位于 `mooncake-store/conf/` 目录下。

**JSON 格式示例** (`master.json`):

```json
{
  "enable_metric_reporting": true,
  "metrics_port": 9003,
  "rpc_port": 50051,
  "rpc_thread_num": 4,
  "rpc_address": "0.0.0.0",
  "rpc_conn_timeout_seconds": 0,
  "rpc_enable_tcp_no_delay": true,
  "default_kv_lease_ttl": 5000,
  "default_kv_soft_pin_ttl": 1800000,
  "allow_evict_soft_pinned_objects": true,
  "eviction_ratio": 0.1,
  "eviction_high_watermark_ratio": 1.0,
  "enable_ha": false,
  "etcd_endpoints": "http://localhost:2379",
  "root_fs_dir": "",
  "cluster_id": "mooncake_cluster",
  "memory_allocator": "offset",
  "client_live_ttl_sec": 60,
  "enable_http_metadata_server": false,
  "http_metadata_server_host": "0.0.0.0",
  "http_metadata_server_port": 8080,
  "enable_offload": false,
  "put_start_discard_timeout_sec": 30,
  "put_start_release_timeout_sec": 600,
  "enable_disk_eviction": true,
  "quota_bytes": 0
}
```

**YAML 格式示例** (`master.yaml`):

```yaml
enable_metric_reporting: true
metrics_port: 9003
rpc_port: 50051
rpc_thread_num: 4
rpc_address: "0.0.0.0"
rpc_conn_timeout_seconds: 0
rpc_enable_tcp_no_delay: true

default_kv_lease_ttl: 5000
default_kv_soft_pin_ttl: 1800000
allow_evict_soft_pinned_objects: true
eviction_ratio: 0.1
eviction_high_watermark_ratio: 1.0

enable_ha: false
etcd_endpoints: "http://localhost:2379"
root_fs_dir: ""
cluster_id: "mooncake_cluster"
memory_allocator: "offset"
client_live_ttl_sec: 60

enable_http_metadata_server: false
http_metadata_server_host: "0.0.0.0"
http_metadata_server_port: 8080
```

### 13.2.2 完整字段说明

#### RPC 与网络配置

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `rpc_port` | `uint32` | 50051 | 是 | Master RPC 服务监听端口 |
| `rpc_address` | `string` | `"0.0.0.0"` | 否 | RPC 服务绑定地址 |
| `rpc_thread_num` | `uint32` | 4 | 是 | RPC 处理线程数 |
| `rpc_conn_timeout_seconds` | `int32` | 0 | 否 | 客户端连接超时（秒），0 表示无限等待 |
| `rpc_enable_tcp_no_delay` | `bool` | `true` | 否 | 是否启用 TCP_NODELAY 以降低延迟 |

#### 监控与指标配置

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `enable_metric_reporting` | `bool` | `true` | 是 | 是否启用 Prometheus 指标上报 |
| `metrics_port` | `uint32` | 9003 | 是 | HTTP 指标端口 |

#### KV 存储与驱逐配置

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `default_kv_lease_ttl` | `uint64` | 5000 | 是 | KV 对象默认租约 TTL（毫秒） |
| `default_kv_soft_pin_ttl` | `uint64` | 1800000 | 否 | 软固定 TTL（毫秒），默认 30 分钟 |
| `allow_evict_soft_pinned_objects` | `bool` | `true` | 是 | 是否允许驱逐软固定对象 |
| `eviction_ratio` | `double` | 0.05 | 是 | 每次驱逐清理的容量比例 |
| `eviction_high_watermark_ratio` | `double` | 0.95 | 是 | 触发驱逐的容量水位线 |
| `client_live_ttl_sec` | `int64` | 10 | 是 | 客户端心跳 TTL（秒），超时后需重新挂载 |

#### 写入超时配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `put_start_discard_timeout_sec` | `uint64` | 30 | Put 操作未启动传输的丢弃超时（秒） |
| `put_start_release_timeout_sec` | `uint64` | 600 | Put 操作未完成的释放超时（秒），默认 10 分钟 |

#### 高可用与 etcd 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_ha` | `bool` | `false` | 是否启用高可用模式（需要 etcd） |
| `etcd_endpoints` | `string` | `"0.0.0.0:2379"` | etcd 集群端点地址 |
| `cluster_id` | `string` | `"mooncake_cluster"` | 集群标识符 |

#### 存储后端配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_offload` | `bool` | `false` | 是否启用数据卸载到持久化存储 |
| `root_fs_dir` | `string` | `""` | 分布式文件系统根目录 |
| `global_file_segment_size` | `int64` | `INT64_MAX` | 全局文件段大小限制（字节） |
| `memory_allocator` | `string` | `"offset"` | 内存分配器类型，可选 `"offset"` 或 `"cachelib"` |
| `enable_disk_eviction` | `bool` | `true` | 是否启用磁盘驱逐 |
| `quota_bytes` | `uint64` | 0 | 存储配额（字节），0 表示无限制 |

#### HTTP 元数据服务

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_http_metadata_server` | `bool` | `false` | 是否启用 HTTP 元数据服务器 |
| `http_metadata_server_host` | `string` | `"0.0.0.0"` | HTTP 元数据服务绑定地址 |
| `http_metadata_server_port` | `uint32` | 8080 | HTTP 元数据服务端口 |

---

## 13.3 Transfer Engine 配置

### 13.3.1 GlobalConfig 结构

Transfer Engine 的核心配置通过 `GlobalConfig` 结构体定义，所有参数均可通过环境变量覆盖。

```cpp
// mooncake-transfer-engine/include/config.h
struct GlobalConfig {
    size_t num_cq_per_ctx = 1;           // 每个上下文的 CQ 数量
    size_t num_comp_channels_per_ctx = 1; // 每个上下文的完成通道数
    uint8_t port = 1;                     // IB 端口号
    int gid_index = -1;                   // GID 索引，-1 表示自动选择
    uint64_t max_mr_size = 0x10000000000; // 最大 MR 大小（约 1TB）
    size_t max_cqe = 4096;               // 每个 CQ 的最大 CQE 数
    int max_ep_per_ctx = 65536;           // 每个上下文的最大端点数
    size_t num_qp_per_ep = 2;            // 每个端点的 QP 数量
    size_t max_sge = 4;                   // 最大 SGE 数量
    size_t max_wr = 256;                  // 最大 WR 数量
    size_t max_inline = 64;               // 最大内联数据大小（字节）
    ibv_mtu mtu_length = IBV_MTU_4096;   // IB MTU 大小
    uint16_t handshake_port = 12001;     // 握手端口
    int workers_per_ctx = 2;             // 每个上下文的 Worker 线程数
    size_t slice_size = 65536;           // 数据分片大小（字节）
    int retry_cnt = 9;                   // 重试次数
    int handshake_listen_backlog = 128;  // 握手监听队列长度
    bool metacache = true;               // 是否启用元数据缓存
    int64_t slice_timeout = -1;          // 分片超时（秒），-1 表示无限
    uint16_t rpc_min_port = 15000;       // RPC 端口范围下限
    uint16_t rpc_max_port = 17000;       // RPC 端口范围上限
    bool use_ipv6 = false;               // 是否使用 IPv6
    size_t fragment_limit = 16384;       // 分片碎片限制
    bool enable_dest_device_affinity = false; // 目标设备亲和性
    int parallel_reg_mr = -1;            // 并行 MR 注册模式
    size_t eic_max_block_size = 64MB;    // EIC 最大块大小
    EndpointStoreType endpoint_store_type = SIEVE; // 端点缓存策略
    int ib_traffic_class = -1;           // IB 流量类别
    int ib_pci_relaxed_ordering_mode = 0; // PCIe Relaxed Ordering 模式
};
```

### 13.3.2 RDMA 参数对照表

| 参数 | 环境变量 | 默认值 | 有效范围 | 说明 |
|------|----------|--------|----------|------|
| `num_cq_per_ctx` | `MC_NUM_CQ_PER_CTX` | 1 | 1-255 | 每个上下文的完成队列数 |
| `num_comp_channels_per_ctx` | `MC_NUM_COMP_CHANNELS_PER_CTX` | 1 | 1-255 | 每上下文完成通道数 |
| `port` | `MC_IB_PORT` | 1 | 0-255 | InfiniBand 端口号 |
| `gid_index` | `MC_GID_INDEX` | -1（自动） | 0-255 | GID 表索引，也支持 `NCCL_IB_GID_INDEX` |
| `max_cqe` | `MC_MAX_CQE_PER_CTX` | 4096 | 1-65535 | 每个 CQ 最大完成事件数 |
| `max_ep_per_ctx` | `MC_MAX_EP_PER_CTX` | 65536 | 1-65535 | 每上下文最大端点数 |
| `num_qp_per_ep` | `MC_NUM_QP_PER_EP` | 2 | 1-255 | 每端点队列对数 |
| `max_sge` | `MC_MAX_SGE` | 4 | 1-65535 | 最大 Scatter/Gather 元素数 |
| `max_wr` | `MC_MAX_WR` | 256 | 1-65535 | 最大 Work Request 队列深度 |
| `max_inline` | `MC_MAX_INLINE` | 64 | 0-65535 | 最大内联数据大小（字节） |
| `mtu_length` | `MC_MTU` | 4096 | 512/1024/2048/4096 | IB 路径 MTU |
| `workers_per_ctx` | `MC_WORKERS_PER_CTX` | 2 | 1-8 | 每上下文 Worker 线程数 |
| `slice_size` | `MC_SLICE_SIZE` | 65536 | >0 | 数据传输分片大小（字节） |
| `retry_cnt` | `MC_RETRY_CNT` | 9 | 1-127 | 传输失败重试次数 |
| `handshake_port` | `MC_HANDSHAKE_PORT` | 12001 | 1-65535 | P2P 握手端口 |
| `slice_timeout` | `MC_SLICE_TIMEOUT` | -1 | 1-65535 | 分片操作超时（秒） |
| `ib_traffic_class` | `MC_IB_TC` | -1 | 0-255 | InfiniBand 流量类别 |
| `ib_pci_relaxed_ordering_mode` | `MC_IB_PCI_RELAXED_ORDERING` | 0 | 0/1/2 | PCIe Relaxed Ordering，0=关闭，1=开启，2=自动 |
| `endpoint_store_type` | `MC_ENDPOINT_STORE_TYPE` | `SIEVE` | FIFO/SIEVE | 端点缓存淘汰策略 |
| `fragment_limit` | `MC_FRAGMENT_RATIO` | slice/4 | >0 且 <slice_size | 碎片比率（fragment_limit = slice_size / ratio） |
| `parallel_reg_mr` | `MC_ENABLE_PARALLEL_REG_MR` | -1 | -1/0/1 | 并行 MR 注册模式 |

### 13.3.3 RPC 与网络配置

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| RPC 端口下限 | `MC_MIN_PRC_PORT` | 15000 | RPC 端口随机选择范围的下界 |
| RPC 端口上限 | `MC_MAX_PRC_PORT` | 17000 | RPC 端口随机选择范围的上界 |
| IPv6 模式 | `MC_USE_IPV6` | 未设置（false） | 设置后启用 IPv6 |
| TCP 绑定地址 | `MC_TCP_BIND_ADDRESS` | 自动检测 | 指定 TCP 绑定 IP 地址 |
| 握手监听队列 | `MC_HANDSHAKE_LISTEN_BACKLOG` | 128 | TCP 监听 backlog 大小 |
| RPC 协议 | `MC_RPC_PROTOCOL` | `"tcp"` | RPC 传输协议，可选 `tcp` 或 `rdma` |

---

## 13.4 TENT 配置

### 13.4.1 transfer-engine.json

TENT（Transfer Engine Next）使用 JSON 配置文件来控制传输行为。配置文件位于 `mooncake-transfer-engine/tent/config/transfer-engine.json`。

```json
{
    "local_segment_name": "",
    "metadata_type": "p2p",
    "metadata_servers": "127.0.0.1:2379",
    "rpc_server_hostname": "127.0.0.1",
    "rpc_server_port": 0,
    "topology": {
        "rdma_whitelist": ["mlx5_0", "mlx5_2"],
        "rdma_blacklist": []
    },
    "log_level": "warning",
    "transports": {
        "rdma": {
            "enable": true,
            "shared_quota_shm_path": "mooncake_quota_shm",
            "max_timeout_ns": 10000000000,
            "device": {
                "num_cq_list": 1,
                "num_comp_channels": 1,
                "port": 1,
                "gid_index": 0,
                "max_cqe": 4096
            },
            "endpoint": {
                "endpoint_store_cap": 256,
                "qp_mul_factor": 1,
                "max_sge": 4,
                "max_qp_wr": 256,
                "max_inline_bytes": 64,
                "path_mtu": 4096
            },
            "workers": {
                "num_workers": 1,
                "max_retry_count": 8,
                "block_size": 65536,
                "grace_period_ns": 50000,
                "rail_topo_path": "/path/to/rail_topo.json"
            }
        },
        "gds": {
            "enable": true
        },
        "shm": {
            "enable": true,
            "cxl_mount_path": "",
            "async_memcpy_threshold": 4
        },
        "mnnvl": {
            "enable": false
        }
    }
}
```

### 13.4.2 TENT 全局参数

| JSON 路径 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `local_segment_name` | `string` | `""` | 本地段名称，通常为节点标识 |
| `metadata_type` | `string` | `"p2p"` | 元数据服务类型，可选 `p2p`、`etcd`、`http`、`redis` |
| `metadata_servers` | `string` | `"127.0.0.1:2379"` | 元数据服务器地址 |
| `rpc_server_hostname` | `string` | `"127.0.0.1"` | RPC 服务器绑定主机名 |
| `rpc_server_port` | `int` | 0 | RPC 服务器端口，0 表示自动分配 |
| `log_level` | `string` | `"warning"` | 日志级别，可选 `info`、`warning`、`error` |

### 13.4.3 TENT 拓扑配置

| JSON 路径 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `topology/rdma_whitelist` | `string[]` | `[]` | RDMA 设备白名单 |
| `topology/rdma_blacklist` | `string[]` | `[]` | RDMA 设备黑名单 |

### 13.4.4 TENT RDMA Device 参数

| JSON 路径 | 类型 | 默认值（推荐） | 说明 |
|-----------|------|----------------|------|
| `transports/rdma/enable` | `bool` | `true` | 是否启用 RDMA 传输 |
| `transports/rdma/shared_quota_shm_path` | `string` | `"mooncake_quota_shm"` | 共享配额的共享内存路径 |
| `transports/rdma/max_timeout_ns` | `uint64` | 10000000000 | 最大超时（纳秒），默认 10 秒 |
| `transports/rdma/device/num_cq_list` | `int` | 6 | CQ 数量，建议等于 `num_workers` |
| `transports/rdma/device/num_comp_channels` | `int` | 1 | 完成通道数 |
| `transports/rdma/device/port` | `uint8` | 1 | IB 端口号 |
| `transports/rdma/device/gid_index` | `int` | 0 | GID 索引 |
| `transports/rdma/device/max_cqe` | `int` | 4096 | 每个 CQ 最大完成事件数 |

### 13.4.5 TENT RDMA Endpoint 参数

| JSON 路径 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `transports/rdma/endpoint/endpoint_store_cap` | `int` | 65536 | 端点缓存容量 |
| `transports/rdma/endpoint/qp_mul_factor` | `int` | 6 | QP 倍数因子，建议等于 `num_workers` |
| `transports/rdma/endpoint/max_sge` | `int` | 4 | 最大 Scatter/Gather 元素数 |
| `transports/rdma/endpoint/max_qp_wr` | `int` | 256 | 最大 QP Work Request 深度 |
| `transports/rdma/endpoint/max_inline_bytes` | `int` | 64 | 最大内联字节数 |
| `transports/rdma/endpoint/path_mtu` | `int` | 4096 | 路径 MTU 大小 |

**高级 QP 参数**（通常无需修改）:

| JSON 路径 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `endpoint/pkey_index` | `uint16` | 0 | 分区键索引 |
| `endpoint/hop_limit` | `uint8` | 16 | 路由跳数限制 |
| `endpoint/flow_label` | `uint32` | 0 | 流标签 |
| `endpoint/traffic_class` | `uint8` | 0 | 流量类别（QoS） |
| `endpoint/service_level` | `uint8` | 0 | 服务级别 |
| `endpoint/max_dest_rd_atomic` | `uint8` | 16 | 目标最大 RDMA 原子操作数 |
| `endpoint/min_rnr_timer` | `uint8` | 12 | RNR NAK 最小重传定时器 |
| `endpoint/send_timeout` | `uint8` | 14 | 发送超时 |
| `endpoint/send_retry_count` | `uint8` | 7 | 发送重试次数 |
| `endpoint/send_rnr_count` | `uint8` | 7 | RNR 重试次数 |
| `endpoint/max_rd_atomic` | `uint8` | 16 | 最大 RDMA 原子操作数 |

### 13.4.6 TENT RDMA Worker 参数

| JSON 路径 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `transports/rdma/workers/num_workers` | `int` | 6 | Worker 线程数量 |
| `transports/rdma/workers/max_retry_count` | `int` | 8 | 传输失败最大重试次数 |
| `transports/rdma/workers/block_size` | `int` | 65536 | 数据传输块大小（字节） |
| `transports/rdma/workers/grace_period_ns` | `uint64` | 5000000 | 优雅等待周期（纳秒），默认 5ms |
| `transports/rdma/workers/rail_topo_path` | `string` | `""` | Rail 拓扑文件路径 |
| `transports/rdma/workers/show_latency_info` | `bool` | `false` | 是否显示延迟信息 |

### 13.4.7 TENT 其他传输后端

| JSON 路径 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `transports/gds/enable` | `bool` | `true` | GPUDirect Storage 支持 |
| `transports/shm/enable` | `bool` | `true` | 共享内存传输 |
| `transports/shm/cxl_mount_path` | `string` | `""` | CXL 设备挂载路径 |
| `transports/shm/async_memcpy_threshold` | `int` | 4 | 异步 memcpy 阈值 |
| `transports/mnnvl/enable` | `bool` | `false` | Multi-Node NVLink 支持 |

### 13.4.8 cluster-topology.json

集群拓扑文件描述节点间 RDMA 通信路径的带宽和延迟信息，用于 TENT 的智能路径选择。

```json
[
  {
    "src_host": "host_uuid_a",
    "dst_host": "host_uuid_b",
    "endpoints": [
      {
        "src_dev": "mlx5_bond_0",
        "dst_dev": "mlx5_bond_1",
        "src_numa": 1,
        "dst_numa": 0,
        "bandwidth": 21182.64,
        "latency": 10.76
      }
    ],
    "partition_matchings": {
      "1-0": [
        {
          "src_dev": "mlx5_bond_0",
          "dst_dev": "mlx5_bond_8",
          "src_numa": 1,
          "dst_numa": 0,
          "bandwidth": 21042.53,
          "latency": 5.92
        }
      ]
    }
  }
]
```

**拓扑文件字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `src_host` | `string` | 源主机 UUID |
| `dst_host` | `string` | 目标主机 UUID |
| `endpoints[].src_dev` | `string` | 源 RDMA 设备名称 |
| `endpoints[].dst_dev` | `string` | 目标 RDMA 设备名称 |
| `endpoints[].src_numa` | `int` | 源 NUMA 节点 ID |
| `endpoints[].dst_numa` | `int` | 目标 NUMA 节点 ID |
| `endpoints[].bandwidth` | `float` | 带宽（MB/s） |
| `endpoints[].latency` | `float` | 延迟（微秒） |
| `partition_matchings` | `object` | NUMA 分区间最优路径匹配，键格式为 `"src_numa-dst_numa"` |

### 13.4.9 TENT 环境变量

| 环境变量 | 说明 |
|----------|------|
| `MC_USE_TENT` | 设置后启用 TENT 引擎（替代传统 Transfer Engine） |
| `MC_USE_TEV1` | 同 `MC_USE_TENT`，别名 |
| `MC_TENT_CONF` | 直接传入 TENT JSON 配置字符串 |
| `MC_IB_PORT` | 向后兼容 - 覆盖 TENT RDMA 设备端口 |
| `MC_GID_INDEX` | 向后兼容 - 覆盖 TENT RDMA GID 索引 |

---

## 13.5 客户端配置 (MooncakeConfig)

### 13.5.1 Python 客户端配置

Python 客户端通过 `MooncakeConfig` 类加载配置，支持 JSON 文件和环境变量两种方式。

**JSON 配置文件**:

```json
{
    "local_hostname": "localhost",
    "metadata_server": "localhost:8080",
    "global_segment_size": 3355443200,
    "local_buffer_size": 1073741824,
    "protocol": "tcp",
    "device_name": "",
    "master_server_address": "localhost:8081"
}
```

**配置字段说明**:

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `local_hostname` | `string` | - | 是 | 本地主机名或 IP |
| `metadata_server` | `string` | - | 是 | 元数据服务器地址 |
| `global_segment_size` | `int` | 3355443200（约 3.125 GiB） | 否 | 全局段大小（字节），支持 `"3GB"` 格式 |
| `local_buffer_size` | `int` | 1073741824（1 GiB） | 否 | 本地缓冲区大小（字节） |
| `protocol` | `string` | `"tcp"` | 否 | 通信协议，可选 `tcp`、`rdma` |
| `device_name` | `string` | `""` | 否 | RDMA 设备名称，如 `mlx5_0` |
| `master_server_address` | `string` | - | 是 | Master 服务地址 |

### 13.5.2 Python 客户端环境变量

| 环境变量 | 对应配置字段 | 默认值 | 说明 |
|----------|-------------|--------|------|
| `MOONCAKE_CONFIG_PATH` | - | 无 | JSON 配置文件路径（优先级最高） |
| `MOONCAKE_MASTER` | `master_server_address` | 无 | Master 服务地址 |
| `MOONCAKE_LOCAL_HOSTNAME` | `local_hostname` | `"localhost"` | 本地主机名 |
| `MOONCAKE_TE_META_DATA_SERVER` | `metadata_server` | `"P2PHANDSHAKE"` | 元数据服务器地址 |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `global_segment_size` | 3355443200 | 全局段大小 |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | `local_buffer_size` | 1073741824 | 本地缓冲区大小 |
| `MOONCAKE_PROTOCOL` | `protocol` | `"tcp"` | 通信协议 |
| `MOONCAKE_DEVICE` | `device_name` | `""` | RDMA 设备名 |

> **加载优先级**: `MOONCAKE_CONFIG_PATH`（指向 JSON 文件） > `MOONCAKE_MASTER` 及其他单独的环境变量。

---

## 13.6 完整环境变量参考

以下是 Mooncake 全部环境变量的汇总表。

### 13.6.1 Transfer Engine - RDMA 参数

| 环境变量 | 默认值 | 有效范围 | 说明 |
|----------|--------|----------|------|
| `MC_NUM_CQ_PER_CTX` | 1 | 1-255 | 每上下文 CQ 数量 |
| `MC_NUM_COMP_CHANNELS_PER_CTX` | 1 | 1-255 | 每上下文完成通道数 |
| `MC_IB_PORT` | 1 | 0-255 | InfiniBand 端口 |
| `MC_GID_INDEX` | -1（自动） | 0-255 | GID 索引 |
| `MC_MAX_CQE_PER_CTX` | 4096 | 1-65535 | 每 CQ 最大完成事件数 |
| `MC_MAX_EP_PER_CTX` | 65536 | 1-65535 | 每上下文最大端点数 |
| `MC_NUM_QP_PER_EP` | 2 | 1-255 | 每端点 QP 数 |
| `MC_MAX_SGE` | 4 | 1-65535 | 最大 SGE 数 |
| `MC_MAX_WR` | 256 | 1-65535 | 最大 WR 队列深度 |
| `MC_MAX_INLINE` | 64 | 0-65535 | 最大内联数据（字节） |
| `MC_MTU` | 4096 | 512/1024/2048/4096 | IB MTU 大小 |
| `MC_IB_TC` | -1 | 0-255 | IB 流量类别 |
| `MC_IB_PCI_RELAXED_ORDERING` | 0 | 0/1/2 | PCIe Relaxed Ordering |

### 13.6.2 Transfer Engine - 传输控制

| 环境变量 | 默认值 | 有效范围 | 说明 |
|----------|--------|----------|------|
| `MC_SLICE_SIZE` | 65536 | >0 | 数据分片大小（字节） |
| `MC_SLICE_TIMEOUT` | -1 | 1-65535 | 分片超时（秒） |
| `MC_WORKERS_PER_CTX` | 2 | 1-8 | 每上下文 Worker 数 |
| `MC_RETRY_CNT` | 9 | 1-127 | 传输重试次数 |
| `MC_FRAGMENT_RATIO` | 4 | >0 | 碎片比率 |
| `MC_MIN_REG_SIZE` | 64MB | >0 | 最小内存注册块大小 |
| `MC_ENDPOINT_STORE_TYPE` | `SIEVE` | FIFO/SIEVE | 端点缓存淘汰策略 |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | 未设置 | 设置即启用 | 目标设备 NUMA 亲和 |
| `MC_ENABLE_PARALLEL_REG_MR` | -1 | -1/0/1 | 并行 MR 注册 |
| `MC_PATH_ROUNDROBIN` | 未设置 | 设置即启用 | RDMA 路径 Round Robin 选择 |
| `MC_DISABLE_METACACHE` | 未设置 | 设置即禁用 | 禁用元数据缓存 |

### 13.6.3 Transfer Engine - 网络与 RPC

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MC_HANDSHAKE_PORT` | 12001 | P2P 握手端口 |
| `MC_HANDSHAKE_LISTEN_BACKLOG` | 128 | 握手监听 backlog |
| `MC_TCP_BIND_ADDRESS` | 自动检测 | TCP 绑定地址 |
| `MC_LEGACY_RPC_PORT_BINDING` | 未设置 | 使用传统 RPC 端口绑定 |
| `MC_MIN_PRC_PORT` | 15000 | RPC 端口下限 |
| `MC_MAX_PRC_PORT` | 17000 | RPC 端口上限 |
| `MC_USE_IPV6` | 未设置 | 启用 IPv6 |
| `MC_RPC_PROTOCOL` | `"tcp"` | RPC 协议，可选 `tcp` 或 `rdma` |

### 13.6.4 Transfer Engine - 传输后端控制

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MC_FORCE_TCP` | 未设置 | 强制使用 TCP 传输 |
| `MC_FORCE_HCA` | 未设置 | 强制使用 HCA/RDMA 传输 |
| `MC_FORCE_MNNVL` | 未设置 | 强制使用 Multi-Node NVLink |
| `MC_USE_NVLINK_IPC` | 未设置 | 使用 NVLink IPC 模式 |
| `MC_USE_HIP_IPC` | 未设置 | 使用 HIP IPC 模式 |
| `MC_CXL_DEV_PATH` | 未设置 | CXL 设备路径 |
| `MC_CXL_DEV_SIZE` | 未设置 | CXL 设备大小 |
| `MC_CUSTOM_TOPO_JSON` | 未设置 | 自定义拓扑 JSON 文件路径 |

### 13.6.5 Transfer Engine - 日志与监控

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MC_LOG_LEVEL` | `INFO` | 日志级别，可选 `TRACE`/`INFO`/`WARNING`/`ERROR` |
| `MC_LOG_DIR` | 未设置 | 日志输出目录，未设置时输出到 stderr |
| `MC_TE_METRIC` | 未设置 | 设置为 `1` 启用 TE 指标上报 |
| `MC_TE_METRIC_INTERVAL_SECONDS` | 未设置 | TE 指标上报间隔（秒） |

### 13.6.6 TENT 引擎

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MC_USE_TENT` | 未设置 | 启用 TENT 引擎 |
| `MC_USE_TEV1` | 未设置 | 同 `MC_USE_TENT` |
| `MC_TENT_CONF` | 未设置 | TENT JSON 配置字符串 |

### 13.6.7 Mooncake Store

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MC_STORE_USE_HUGEPAGE` | 未设置 | 启用大页内存 |
| `MC_STORE_HUGEPAGE_SIZE` | 系统默认 | 大页大小（字节） |
| `MC_STORE_MEMCPY` | 未设置（禁用） | 启用 memcpy 操作 |
| `MC_STORE_CLUSTER_ID` | 未设置 | 存储集群 ID |
| `MC_STORE_CLIENT_METRIC` | `1`（启用） | 客户端指标上报开关 |
| `MC_STORE_CLIENT_METRIC_INTERVAL` | 未设置 | 客户端指标上报间隔（秒） |
| `MC_MS_AUTO_DISC` | 未设置 | Master 自动发现，`0`=关闭，`1`=开启 |
| `MC_MS_FILTERS` | 未设置 | Master 发现过滤器 |

### 13.6.8 Mooncake Offload 存储

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR` | 未设置 | 存储后端描述符 |
| `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` | 未设置 | 文件存储路径 |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | 未设置 | 本地缓冲区大小（字节） |
| `MOONCAKE_OFFLOAD_FSDIR` | 未设置 | 卸载文件系统目录 |
| `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT` | 未设置 | 桶内键数量限制 |
| `MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES` | 未设置 | 桶大小限制（字节） |
| `MOONCAKE_OFFLOAD_TOTAL_KEYS_LIMIT` | 未设置 | 总键数量限制 |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | 未设置 | 总大小限制（字节） |
| `MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS` | 未设置 | 心跳间隔（秒） |
| `MOONCAKE_SCANMETA_ITERATOR_KEYS_LIMIT` | 未设置 | 扫描元数据迭代器键限制 |

### 13.6.9 Python 客户端 (mooncake-wheel)

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MOONCAKE_CONFIG_PATH` | 未设置 | JSON 配置文件路径 |
| `MOONCAKE_MASTER` | 未设置 | Master 服务地址 |
| `MOONCAKE_LOCAL_HOSTNAME` | `"localhost"` | 本地主机名 |
| `MOONCAKE_TE_META_DATA_SERVER` | `"P2PHANDSHAKE"` | 元数据服务地址 |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | 3355443200 | 全局段大小 |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | 1073741824 | 本地缓冲区大小 |
| `MOONCAKE_PROTOCOL` | `"tcp"` | 通信协议 |
| `MOONCAKE_DEVICE` | `""` | RDMA 设备名 |
| `MC_METADATA_SERVER` | `"127.0.0.1:2379"` | 元数据服务地址（测试用） |

---

## 13.7 默认值速查表

以下汇总所有关键配置项及其默认值，方便快速查阅。

### MasterService 默认值

| 配置项 | 默认值 | 单位 |
|--------|--------|------|
| RPC 端口 | 50051 | - |
| RPC 线程数 | 4 | - |
| 指标端口 | 9003 | - |
| KV 租约 TTL | 5000 | 毫秒 |
| 软固定 TTL | 1800000 | 毫秒（30 分钟） |
| 驱逐比率 | 0.05 (5%) | - |
| 高水位线 | 0.95 (95%) | - |
| 客户端心跳 TTL | 10 | 秒 |
| Put 丢弃超时 | 30 | 秒 |
| Put 释放超时 | 600 | 秒（10 分钟） |
| etcd 端点 | `0.0.0.0:2379` | - |
| 集群 ID | `mooncake_cluster` | - |

### Transfer Engine RDMA 默认值

| 配置项 | 默认值 | 单位 |
|--------|--------|------|
| CQ 数量 | 1 | 每上下文 |
| QP 数量 | 2 | 每端点 |
| 最大 SGE | 4 | - |
| 最大 WR | 256 | - |
| 最大内联 | 64 | 字节 |
| MTU | 4096 | 字节 |
| 分片大小 | 65536 | 字节（64KB） |
| Worker 数 | 2 | 每上下文 |
| 重试次数 | 9 | - |
| 握手端口 | 12001 | - |

### TENT RDMA 默认值

| 配置项 | 默认值 | 单位 |
|--------|--------|------|
| CQ 数量 | 6 | 每设备 |
| QP 倍数因子 | 6 | - |
| 最大 SGE | 4 | - |
| 最大 QP WR | 256 | - |
| 块大小 | 65536 | 字节（64KB） |
| Worker 数 | 6 | - |
| 优雅等待周期 | 5000000 | 纳秒（5ms） |
| 端点缓存容量 | 65536 | - |

### Python 客户端默认值

| 配置项 | 默认值 | 单位 |
|--------|--------|------|
| 全局段大小 | 3355443200 | 字节（约 3.125 GiB） |
| 本地缓冲区 | 1073741824 | 字节（1 GiB） |
| 协议 | `tcp` | - |
| 元数据服务 | `P2PHANDSHAKE` | - |
