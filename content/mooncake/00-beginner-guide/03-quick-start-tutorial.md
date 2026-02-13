---
title: "快速上手教程"
weight: 3
---

# 快速上手教程

[上一篇: 核心概念详解](02-core-concepts.md) | [目录](../README.md) | [下一篇: 编译与开发指南](04-build-and-development.md)

---

> 本篇将引导你完成 Mooncake 的安装和基础使用，包括 Transfer Engine 和 Mooncake Store 的实际运行示例。

---

## 1. 前置条件

### 1.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | x86_64 / ARM64 | 多核 CPU |
| 内存 | 8 GB | 32 GB+ |
| 网络 | 普通以太网（TCP 模式） | RDMA 网卡（InfiniBand / RoCE） |
| GPU | 可选 | NVIDIA GPU + CUDA 12.1+（使用 GPU Direct 时需要） |
| 存储 | 10 GB 可用空间 | NVMe SSD（使用 SSD 缓存层时需要） |

> **提示**: Mooncake 同时支持 TCP 和 RDMA 两种传输协议。如果没有 RDMA 网卡，可以使用 TCP 模式进行开发和测试，功能完全一致，仅传输性能有差异。

### 1.2 操作系统

| 操作系统 | 支持状态 |
|---------|---------|
| Ubuntu 22.04 LTS+ | 推荐 |
| CentOS / Alibaba Linux | 支持 |
| 其他 Linux 发行版 | 需自行验证 |
| macOS / Windows | 不支持 |

### 1.3 软件依赖

- Python 3.8+
- pip（用于安装 PyPI 包）
- etcd 或 HTTP 元数据服务（二选一）

---

## 2. 安装方式

### 2.1 方式一 - 通过 PyPI 安装（推荐）

这是最快的安装方式，适合大多数用户：

**有 CUDA 环境的系统:**

```bash
pip install mooncake-transfer-engine
```

> 该包包含 Mooncake Transfer Engine 和 Mooncake Store 的 Python 绑定，要求 CUDA 12.1+，同时支持 Mooncake-EP 和 GPU 拓扑检测功能。

**无 CUDA 环境的系统:**

```bash
pip install mooncake-transfer-engine-non-cuda
```

> 不包含 GPU 相关功能，适用于没有 CUDA 依赖的开发和测试环境。

### 2.2 方式二 - 从源码编译

如果需要定制编译选项或参与开发，请参考 [编译与开发指南](04-build-and-development.md)。

快速编译步骤：

```bash
# 1. 克隆仓库
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake

# 2. 安装依赖（需要 root 权限）
sudo bash dependencies.sh

# 3. 编译
mkdir build && cd build
cmake ..
make -j$(nproc)

# 4. 安装
sudo make install
```

### 2.3 验证安装

```bash
python3 -c "from mooncake.engine import TransferEngine; print('Transfer Engine 导入成功')"
python3 -c "from mooncake.store import MooncakeDistributedStore; print('Mooncake Store 导入成功')"
```

如果两条命令都正常输出，说明安装成功。

---

## 3. 启动元数据服务

Mooncake 需要一个元数据服务来协调各节点之间的信息交换。你可以选择以下几种方式之一：

### 3.1 方式一 - P2P Handshake（最简单，适合快速测试）

在初始化 Transfer Engine 时指定 `P2PHANDSHAKE` 即可，无需额外启动服务：

```python
METADATA_SERVER = "P2PHANDSHAKE"
```

> 这种方式不需要额外的元数据服务，节点之间直接握手交换信息。适合本地开发测试。

### 3.2 方式二 - 内置 HTTP 元数据服务

启动 `mooncake_master` 并开启 HTTP 元数据服务：

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

在代码中使用 HTTP 元数据服务地址：

```python
METADATA_SERVER = "http://localhost:8080/metadata"
```

### 3.3 方式三 - etcd

如果集群中已有 etcd 服务，可以直接使用：

```bash
# 安装并启动 etcd（如果尚未安装）
sudo apt-get install -y etcd
sudo systemctl start etcd
```

在代码中使用 etcd 地址：

```python
METADATA_SERVER = "etcd://localhost:2379"
```

### 3.4 组件启动顺序

```mermaid
graph LR
    A["启动元数据服务"] --> B["启动 Mooncake Master"]
    B --> C["启动 Transfer Engine"]
    C --> D["启动 Mooncake Store"]
    D --> E["运行应用逻辑"]

    style A fill:#4ecdc4,color:#fff
    style B fill:#45b7d1,color:#fff
    style C fill:#96ceb4,color:#fff
    style D fill:#ffeaa7,color:#333
    style E fill:#dfe6e9,color:#333
```

> **注意**: 如果使用 P2P Handshake 模式，则无需步骤 A 和 B，直接从步骤 C 开始。

---

## 4. Transfer Engine 基础示例

### 4.1 示例场景

我们将实现一个最简单的数据传输场景：一个 Receiver（接收端）等待数据，一个 Sender（发送端）将数据写入 Receiver 的内存。

### 4.2 启动 Receiver（Server 端）

创建文件 `receiver.py`：

```python
import numpy as np
import zmq
from mooncake.engine import TransferEngine

def main():
    # 初始化 ZMQ 用于交换 buffer 信息
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:5555")

    HOSTNAME = "localhost"
    METADATA_SERVER = "P2PHANDSHAKE"
    PROTOCOL = "tcp"          # 使用 TCP 协议（无需 RDMA 网卡）
    DEVICE_NAME = ""          # 留空，自动发现设备

    # 初始化 Transfer Engine
    server_engine = TransferEngine()
    server_engine.initialize(HOSTNAME, METADATA_SERVER, PROTOCOL, DEVICE_NAME)
    session_id = f"{HOSTNAME}:{server_engine.get_rpc_port()}"

    # 分配 1MB 接收缓冲区
    server_buffer = np.zeros(1024 * 1024, dtype=np.uint8)
    server_ptr = server_buffer.ctypes.data
    server_len = server_buffer.nbytes

    # RDMA 模式需要注册内存（TCP 模式可跳过）
    if PROTOCOL == "rdma":
        ret = server_engine.register_memory(server_ptr, server_len)
        if ret != 0:
            raise RuntimeError("内存注册失败")

    print(f"Receiver 已启动, Session ID: {session_id}")

    # 将 buffer 信息发送给 Sender
    socket.send_json({
        "session_id": session_id,
        "ptr": server_ptr,
        "len": server_len
    })

    # 等待数据传入
    try:
        while True:
            input("按 Ctrl+C 退出...")
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        if PROTOCOL == "rdma":
            server_engine.unregister_memory(server_ptr)
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
```

### 4.3 启动 Sender（Client 端）

创建文件 `sender.py`：

```python
import numpy as np
import zmq
from mooncake.engine import TransferEngine

def main():
    # 通过 ZMQ 获取 Receiver 的 buffer 信息
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:5555")

    print("等待 Receiver 的 buffer 信息...")
    buffer_info = socket.recv_json()
    server_session_id = buffer_info["session_id"]
    server_ptr = buffer_info["ptr"]
    server_len = buffer_info["len"]
    print(f"已收到 Receiver 信息 - Session ID: {server_session_id}")

    HOSTNAME = "localhost"
    METADATA_SERVER = "P2PHANDSHAKE"
    PROTOCOL = "tcp"
    DEVICE_NAME = ""

    # 初始化 Sender 端 Transfer Engine
    client_engine = TransferEngine()
    client_engine.initialize(HOSTNAME, METADATA_SERVER, PROTOCOL, DEVICE_NAME)

    # 准备发送数据（1MB，填充为全 1）
    client_buffer = np.ones(1024 * 1024, dtype=np.uint8)
    client_ptr = client_buffer.ctypes.data
    client_len = client_buffer.nbytes

    # 注册内存
    ret = client_engine.register_memory(client_ptr, client_len)
    if ret != 0:
        raise RuntimeError("内存注册失败")

    # 执行 10 次数据传输
    print("开始传输数据...")
    for i in range(10):
        ret = client_engine.transfer_sync_write(
            server_session_id,
            client_ptr,
            server_ptr,
            min(client_len, server_len)
        )
        if ret >= 0:
            print(f"第 {i+1} 次传输成功")
        else:
            print(f"第 {i+1} 次传输失败")

    # 清理资源
    client_engine.unregister_memory(client_ptr)
    socket.close()
    context.term()
    print("传输完成")

if __name__ == "__main__":
    main()
```

### 4.4 运行示例

打开两个终端窗口：

```bash
# 终端 1 - 启动 Receiver
python3 receiver.py

# 终端 2 - 启动 Sender
python3 sender.py
```

预期输出（Sender 端）：

```
等待 Receiver 的 buffer 信息...
已收到 Receiver 信息 - Session ID: localhost:12345
开始传输数据...
第 1 次传输成功
第 2 次传输成功
...
第 10 次传输成功
传输完成
```

---

## 5. Mooncake Store 基础示例

### 5.1 前置条件

Mooncake Store 需要先启动 `mooncake_master` 服务：

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

### 5.2 Hello World 示例

```python
from mooncake.store import MooncakeDistributedStore

# 1. 创建 Store 实例
store = MooncakeDistributedStore()

# 2. 初始化配置
store.setup(
    "localhost",                          # 本节点地址
    "http://localhost:8080/metadata",     # HTTP 元数据服务地址
    512 * 1024 * 1024,                    # Segment 大小: 512MB
    128 * 1024 * 1024,                    # 本地缓冲区: 128MB
    "tcp",                                # 传输协议: tcp 或 rdma
    "",                                   # RDMA 设备名（留空自动选择）
    "localhost:50051"                     # Master 服务地址
)

# 3. 存储数据
store.put("hello_key", b"Hello, Mooncake Store!")

# 4. 读取数据
data = store.get("hello_key")
print(data.decode())  # 输出: Hello, Mooncake Store!

# 5. 清理资源
store.close()
```

### 5.3 预期输出

```
Hello, Mooncake Store!
```

### 5.4 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| 节点地址 | 当前节点的主机名或 IP | `"localhost"` |
| 元数据服务地址 | HTTP 元数据服务的 URL | `"http://localhost:8080/metadata"` |
| Segment 大小 | 存储分段大小（字节） | `512 * 1024 * 1024`（512MB） |
| 本地缓冲区大小 | 本地缓冲区大小（字节） | `128 * 1024 * 1024`（128MB） |
| 传输协议 | 数据传输协议 | `"tcp"` 或 `"rdma"` |
| RDMA 设备名 | RDMA 网卡设备名 | `""` 表示自动选择 |
| Master 地址 | mooncake_master 的 gRPC 地址 | `"localhost:50051"` |

---

## 6. 常见问题与排错

### 6.1 安装相关

**问题: `pip install` 失败**

```
ERROR: Could not find a version that satisfies the requirement mooncake-transfer-engine
```

解决方案：
- 确认 Python 版本 >= 3.8
- 确认 pip 已更新到最新版本: `pip install --upgrade pip`
- 如果在国内网络环境，尝试使用镜像源: `pip install mooncake-transfer-engine -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 6.2 元数据服务相关

**问题: 连接元数据服务失败**

```
Failed to connect to metadata server
```

排查步骤：
1. 确认元数据服务已启动（etcd 或 HTTP 服务）
2. 检查网络连通性: `curl http://localhost:8080/metadata`
3. 检查防火墙规则，确保对应端口已开放
4. 如果使用 etcd，确认 etcd 版本兼容

### 6.3 RDMA 相关

**问题: RDMA 设备未找到**

```
Failed to find RDMA device
```

排查步骤：
1. 确认已安装 RDMA 驱动: `ibv_devinfo`
2. 确认 RDMA 设备存在: `ls /dev/infiniband/`
3. 确认 libibverbs 已安装: `dpkg -l | grep libibverbs`
4. 如果没有 RDMA 设备，可切换为 TCP 模式: `PROTOCOL = "tcp"`

**问题: 内存注册失败**

```
Mooncake memory registration failed
```

排查步骤：
1. 检查 `memlock` 限制: `ulimit -l`，建议设置为 `unlimited`
2. 修改限制: `ulimit -l unlimited` 或编辑 `/etc/security/limits.conf`
3. 确认注册的内存地址和大小有效
4. 确认没有重复注册同一块内存

### 6.4 传输相关

**问题: 数据传输失败**

```
Transfer failed!
```

排查步骤：
1. 确认 Receiver 已启动并正在监听
2. 确认 Session ID 正确（通过 ZMQ 或其他方式传递）
3. 检查目标地址和长度是否合法
4. 检查网络连通性
5. 如果使用 RDMA，确认两端的 RDMA 网卡可以互通

### 6.5 性能相关

**问题: TCP 模式下传输速度慢**

这是预期行为。TCP 模式主要用于开发测试，生产环境建议使用 RDMA 模式以获得最佳性能。

| 模式 | 典型带宽 |
|------|---------|
| TCP | 1-10 Gbps |
| RDMA | 100-400 Gbps |

---

## 下一步

成功运行示例后，如果你想深入了解编译选项和开发流程：

> [下一篇: 编译与开发指南](04-build-and-development.md) -- 详解从源码编译 Mooncake 的完整流程、CMake 选项和开发工作流。
