---
title: "模块映射参考"
weight: 2
---

## 概述

本文档提供了从旧模块结构到新章节结构的完整映射表，帮助你快速找到感兴趣的内容。

## 模块映射表

### Module 00: Overview → 00-overview 扬帆起航

| 旧路径 | 新路径 | 内容 |
|--------|--------|------|
| `module-00-overview/00-architecture-overview.md` | [00-overview/01-architecture-overview](/pytorch/00-overview/01-architecture-overview/) | PyTorch 源码架构全景 |
| `module-00-overview/01-build-and-dev-env.md` | [00-overview/02-build-and-dev-env](/pytorch/00-overview/02-build-and-dev-env/) | 开发环境搭建 |
| `module-00-overview/02-key-concepts.md` | [00-overview/03-key-concepts](/pytorch/00-overview/03-key-concepts/) | 核心概念速览 |
| `module-00-overview/03-execution-flow-overview.md` | [00-overview/04-execution-flow-overview](/pytorch/00-overview/04-execution-flow-overview/) | 执行流程概览 |
| `module-00-overview/04-code-navigation-tips.md` | [00-overview/05-code-navigation-tips](/pytorch/00-overview/05-code-navigation-tips/) | 代码导航技巧 |

### Module 01: Tensor & Storage → 多个章节

| 旧路径 | 新路径 | 章节 |
|--------|--------|------|
| `module-01-tensor-storage/00-tensor-basics-tutorial.md` | [01-introduction/01-tensor-basics-tutorial](/pytorch/01-introduction/01-tensor-basics-tutorial/) | 入门篇 |
| `module-01-tensor-storage/01-tensorimpl-deep-dive.md` | [02-fundamentals/01-tensorimpl-deep-dive](/pytorch/02-fundamentals/01-tensorimpl-deep-dive/) | 基础知识 |
| `module-01-tensor-storage/02-storage-and-allocator.md` | [02-fundamentals/02-storage-and-allocator](/pytorch/02-fundamentals/02-storage-and-allocator/) | 基础知识 |
| `module-01-tensor-storage/03-dtype-device-layout.md` | [02-fundamentals/03-dtype-device-layout](/pytorch/02-fundamentals/03-dtype-device-layout/) | 基础知识 |
| `module-01-tensor-storage/04-tensor-creation-codepath.md` | [02-fundamentals/04-tensor-creation-codepath](/pytorch/02-fundamentals/04-tensor-creation-codepath/) | 基础知识 |
| `module-01-tensor-storage/05-symbolic-shapes.md` | [02-fundamentals/07-symbolic-shapes](/pytorch/02-fundamentals/07-symbolic-shapes/) | 基础知识 |

### Module 02: Autograd → 多个章节

| 旧路径 | 新路径 | 章节 |
|--------|--------|------|
| `module-02-autograd/00-autograd-basics-tutorial.md` | [01-introduction/02-autograd-basics-tutorial](/pytorch/01-introduction/02-autograd-basics-tutorial/) | 入门篇 |
| `module-02-autograd/01-computational-graph.md` | [02-fundamentals/05-computational-graph](/pytorch/02-fundamentals/05-computational-graph/) | 基础知识 |
| `module-02-autograd/02-engine-execution.md` | [02-fundamentals/06-engine-execution](/pytorch/02-fundamentals/06-engine-execution/) | 基础知识 |
| `module-02-autograd/03-custom-autograd-function.md` | [04-code-walkthrough/06-custom-autograd-function](/pytorch/04-code-walkthrough/06-custom-autograd-function/) | 代码走读 |
| `module-02-autograd/04-grad-mode-and-inference.md` | [04-code-walkthrough/07-grad-mode-and-inference](/pytorch/04-code-walkthrough/07-grad-mode-and-inference/) | 代码走读 |
| `module-02-autograd/05-backward-codepath-debug.md` | [04-code-walkthrough/01-backward-codepath-debug](/pytorch/04-code-walkthrough/01-backward-codepath-debug/) | 代码走读 |
| `module-02-autograd/06-saved-tensors-hooks.md` | [04-code-walkthrough/08-saved-tensors-hooks](/pytorch/04-code-walkthrough/08-saved-tensors-hooks/) | 代码走读 |

### Module 03: nn.Module → 多个章节

| 旧路径 | 新路径 | 章节 |
|--------|--------|------|
| `module-03-nn-module/00-nn-module-tutorial.md` | [01-introduction/03-nn-module-tutorial](/pytorch/01-introduction/03-nn-module-tutorial/) | 入门篇 |
| `module-03-nn-module/01-module-core-mechanism.md` | [03-core-modules/01-module-core-mechanism](/pytorch/03-core-modules/01-module-core-mechanism/) | 核心模块详解 |
| `module-03-nn-module/02-module-hooks-system.md` | [03-core-modules/02-module-hooks-system](/pytorch/03-core-modules/02-module-hooks-system/) | 核心模块详解 |
| `module-03-nn-module/03-common-layers-analysis.md` | [04-code-walkthrough/03-common-layers-analysis](/pytorch/04-code-walkthrough/03-common-layers-analysis/) | 代码走读 |
| `module-03-nn-module/04-functional-vs-module.md` | [04-code-walkthrough/04-functional-vs-module](/pytorch/04-code-walkthrough/04-functional-vs-module/) | 代码走读 |
| `module-03-nn-module/05-module-serialization.md` | [04-code-walkthrough/05-module-serialization](/pytorch/04-code-walkthrough/05-module-serialization/) | 代码走读 |

### Module 04: Optimizer → 多个章节

| 旧路径 | 新路径 | 章节 |
|--------|--------|------|
| `module-04-optimizer/00-optimizer-tutorial.md` | [01-introduction/04-optimizer-tutorial](/pytorch/01-introduction/04-optimizer-tutorial/) | 入门篇 |
| `module-04-optimizer/01-optimizer-base-class.md` | [03-core-modules/03-optimizer-base-class](/pytorch/03-core-modules/03-optimizer-base-class/) | 核心模块详解 |
| `module-04-optimizer/02-adam-algorithm-analysis.md` | [03-core-modules/04-adam-algorithm-analysis](/pytorch/03-core-modules/04-adam-algorithm-analysis/) | 核心模块详解 |
| `module-04-optimizer/03-lr-scheduler.md` | [03-core-modules/05-lr-scheduler](/pytorch/03-core-modules/05-lr-scheduler/) | 核心模块详解 |

### Module 05: Dispatch → 多个章节

| 旧路径 | 新路径 | 章节 |
|--------|--------|------|
| `module-05-dispatch/01-dispatch-key-system.md` | [03-core-modules/06-dispatch-key-system](/pytorch/03-core-modules/06-dispatch-key-system/) | 核心模块详解 |
| `module-05-dispatch/02-dispatcher-core.md` | [03-core-modules/07-dispatcher-core](/pytorch/03-core-modules/07-dispatcher-core/) | 核心模块详解 |
| `module-05-dispatch/03-operator-registration.md` | [03-core-modules/08-operator-registration](/pytorch/03-core-modules/08-operator-registration/) | 核心模块详解 |
| `module-05-dispatch/04-dispatch-codepath-debug.md` | [04-code-walkthrough/02-dispatch-codepath-debug](/pytorch/04-code-walkthrough/02-dispatch-codepath-debug/) | 代码走读 |
| `module-05-dispatch/05-boxing-and-fallthrough.md` | [04-code-walkthrough/09-boxing-and-fallthrough](/pytorch/04-code-walkthrough/09-boxing-and-fallthrough/) | 代码走读 |

### Module 06: Dynamo → 05-advanced-topics 高级主题

| 旧路径 | 新路径 |
|--------|--------|
| `module-06-dynamo/00-dynamo-tutorial.md` | [05-advanced-topics/04-dynamo-tutorial](/pytorch/05-advanced-topics/04-dynamo-tutorial/) |
| `module-06-dynamo/02-bytecode-analysis.md` | [05-advanced-topics/05-bytecode-analysis](/pytorch/05-advanced-topics/05-bytecode-analysis/) |
| `module-06-dynamo/03-guard-mechanism.md` | [05-advanced-topics/06-guard-mechanism](/pytorch/05-advanced-topics/06-guard-mechanism/) |

### Module 07: FX → 05-advanced-topics 高级主题

| 旧路径 | 新路径 |
|--------|--------|
| `module-07-fx/00-fx-tutorial.md` | [05-advanced-topics/02-fx-tutorial](/pytorch/05-advanced-topics/02-fx-tutorial/) |
| `module-07-fx/01-graph-and-node.md` | [05-advanced-topics/03-graph-and-node](/pytorch/05-advanced-topics/03-graph-and-node/) |

### Module 08: Inductor → 05-advanced-topics 高级主题

| 旧路径 | 新路径 |
|--------|--------|
| `module-08-inductor/00-inductor-overview.md` | [05-advanced-topics/07-inductor-overview](/pytorch/05-advanced-topics/07-inductor-overview/) |
| `module-08-inductor/04-triton-codegen.md` | [05-advanced-topics/08-triton-codegen](/pytorch/05-advanced-topics/08-triton-codegen/) |

### Module 09: Distributed → 05-advanced-topics 高级主题

| 旧路径 | 新路径 |
|--------|--------|
| `module-09-distributed/00-distributed-overview.md` | [05-advanced-topics/09-distributed-overview](/pytorch/05-advanced-topics/09-distributed-overview/) |
| `module-09-distributed/03-ddp-fsdp.md` | [05-advanced-topics/10-ddp-fsdp](/pytorch/05-advanced-topics/10-ddp-fsdp/) |

### Module 10: CUDA & Memory → 05-advanced-topics 高级主题

| 旧路径 | 新路径 |
|--------|--------|
| `module-10-cuda-memory/00-cuda-memory-tutorial.md` | [05-advanced-topics/11-cuda-memory-tutorial](/pytorch/05-advanced-topics/11-cuda-memory-tutorial/) |
| `module-10-cuda-memory/03-amp-mixed-precision.md` | [05-advanced-topics/12-amp-mixed-precision](/pytorch/05-advanced-topics/12-amp-mixed-precision/) |

## 新增内容

以下是新章节结构中新增的内容：

| 文件 | 描述 |
|------|------|
| [05-advanced-topics/01-compiler-stack-overview](/pytorch/05-advanced-topics/01-compiler-stack-overview/) | 编译器栈概览（综合 FX、Dynamo、Inductor） |
| [appendix/02-module-reference](/pytorch/appendix/02-module-reference/) | 模块映射参考（本文档） |
| [appendix/03-glossary](/pytorch/appendix/03-glossary/) | PyTorch 术语表 |
| [appendix/04-code-map](/pytorch/appendix/04-code-map/) | 源码导航地图 |
| [appendix/05-references](/pytorch/appendix/05-references/) | 参考资源和外部链接 |

## 章节组织逻辑

### 为什么这样重组？

1. **入门篇**：集中所有 Tutorial 文件，为新手提供清晰的学习路径
2. **基础知识**：合并 Tensor 和 Autograd 的底层机制，形成完整的基础理论
3. **核心模块详解**：整合 nn.Module、Optimizer、Dispatch 三大核心系统
4. **代码走读**：收集所有 Debug 和分析类文章，方便实战调试
5. **高级主题**：聚焦编译器栈、分布式和性能优化等高级话题

## 快速查找指南

### 按主题查找

- **Tensor 相关**：[入门篇](../01-introduction/01-tensor-basics-tutorial/) → [基础知识](../02-fundamentals/01-tensorimpl-deep-dive/)
- **Autograd 相关**：[入门篇](../01-introduction/02-autograd-basics-tutorial/) → [基础知识](../02-fundamentals/05-computational-graph/) → [代码走读](../04-code-walkthrough/01-backward-codepath-debug/)
- **nn.Module 相关**：[入门篇](../01-introduction/03-nn-module-tutorial/) → [核心模块](../03-core-modules/01-module-core-mechanism/) → [代码走读](../04-code-walkthrough/03-common-layers-analysis/)
- **编译器栈**：[编译器栈概览](../05-advanced-topics/01-compiler-stack-overview/) → [FX](../05-advanced-topics/02-fx-tutorial/) → [Dynamo](../05-advanced-topics/04-dynamo-tutorial/) → [Inductor](../05-advanced-topics/07-inductor-overview/)
- **分布式训练**：[分布式概览](../05-advanced-topics/09-distributed-overview/) → [DDP/FSDP](../05-advanced-topics/10-ddp-fsdp/)

### 按难度查找

- **初学者**：[00-overview](../00-overview/) → [01-introduction](../01-introduction/)
- **进阶者**：[02-fundamentals](../02-fundamentals/) → [03-core-modules](../03-core-modules/)
- **专家级**：[04-code-walkthrough](../04-code-walkthrough/) → [05-advanced-topics](../05-advanced-topics/)

## 反馈

如果你在查找内容时遇到困难，或者对章节组织有建议，欢迎通过以下方式反馈：

- GitHub Issues: [inference-cookbook/issues](https://github.com/SuperMarioYL/inference-cookbook/issues)
- 邮件联系：参见 [参考资源](../05-references/)
