---
title: "PyTorch"
linkTitle: "PyTorch"
weight: 1
description: "PyTorch 源码深度解析 - 从架构到实现的完整指南"
---

PyTorch 是当今最流行的深度学习框架之一，被广泛应用于学术研究和工业生产。本文档深入剖析 PyTorch 源码，帮助开发者理解其内部机制。

## 适合谁阅读

本文档适合以下读者：

- **有 PyTorch 使用经验**，想了解内部实现机制的开发者
- **对编译器技术感兴趣**，想理解 `torch.compile()` 工作原理的工程师
- **深度学习系统研究者**，需要深入理解 Autograd、Dispatch、分布式训练等核心模块

## 你将学到什么

本文档涵盖 PyTorch 的 11 个核心模块：

- **基础架构**：Tensor/Storage、Autograd、NN Module、优化器、Dispatch 系统
- **编译器栈**：FX 图系统、TorchDynamo、TorchInductor、模型导出
- **进阶主题**：分布式训练（DDP/FSDP）、CUDA 内存管理、混合精度训练

每个模块都配有：源码解析、执行链路追踪、可运行示例、Mermaid 架构图。

## 三条阅读路线

我们提供三条学习路径，适配不同背景和目标：

1. **初学者路线** - 从上层到底层，从 NN Module 到 Tensor Storage
2. **编译器路线** - `torch.compile()` 全栈，理解 FX → Dynamo → Inductor
3. **专项路线** - 按需选读分布式训练、GPU 编程、模型部署

详细路线规划请参阅 [阅读路线指南](./READING_GUIDE.md)。

## 快速开始

建议先阅读 [Module 0 - 扬帆起航](./00-overview/) 建立整体认知，了解 PyTorch 的三层架构和核心概念。
