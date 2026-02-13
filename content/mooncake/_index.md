---
title: "Mooncake"
linkTitle: "Mooncake"
weight: 26
description: "Mooncake KVCache 分离式 LLM 推理服务平台深度解析 - FAST 2025 Best Paper"
---

**FAST 2025 Best Paper** | 以存储换计算的 LLM 推理系统架构创新

Mooncake 是由 Moonshot AI (Kimi) 团队研发的 **KVCache-centric 分离式 LLM 推理服务系统**。其核心思想是将 Prefill 与 Decode 阶段解耦，利用集群中所有节点的 CPU 内存和 NVMe SSD 构建 PB 级分布式 KVCache 缓存池，实现跨节点 KVCache 高效共享。

## 核心创新

- **分离式架构**：Prefill 和 Decode 解耦，独立扩展
- **分布式存储**：Mooncake Store 提供 PB 级 KVCache 缓存池
- **高性能传输**：Transfer Engine 基于 RDMA 实现低延迟跨节点传输
- **拓扑感知调度**：TENT 算法优化网络传输路径

## 生产实践成果

在 Kimi 服务的实际生产环境（数千节点）中，Mooncake 相比 vLLM 基准系统在 100ms TBT SLO 约束下实现了 **+498%** 的有效请求容量提升。

## 适合谁阅读

- **LLM 推理系统工程师** - 需要优化推理吞吐量和延迟
- **分布式系统架构师** - 关注大规模 KVCache 管理方案
- **RDMA 网络工程师** - 理解高性能数据传输设计
- **运维人员** - 部署和调优 Mooncake 系统

## 三条学习路径

1. **初学者路径** - 从基础概念入手，理解架构设计和请求流程
2. **进阶路径** - 直接进入核心模块（Store、Transfer Engine）和调度算法
3. **运维路径** - 聚焦部署、监控、性能调优和故障排查

详细路径规划和场景化导航请参阅 [项目概述](./README.md)。

## 快速开始

建议先阅读 [初学者指南](./00-beginner-guide/) 快速了解 Mooncake 的核心价值和设计理念。
