---
title: "参考资源"
weight: 5
---

## 官方资源

### 文档

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html) - 完整的 API 文档和教程
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - 官方教程合集
- [PyTorch C++ API 文档](https://pytorch.org/cppdocs/) - C++ Frontend 文档
- [PyTorch Mobile 文档](https://pytorch.org/mobile/home/) - 移动端部署文档

### 源码与开发

- [PyTorch GitHub](https://github.com/pytorch/pytorch) - 主仓库
- [PyTorch 贡献指南](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md) - 如何参与贡献
- [PyTorch RFC](https://github.com/pytorch/rfcs) - 功能提案和设计文档
- [PyTorch Dev Discussions](https://dev-discuss.pytorch.org/) - 开发者论坛

### 博客与公告

- [PyTorch Blog](https://pytorch.org/blog/) - 官方博客
- [PyTorch Medium](https://medium.com/pytorch) - PyTorch Medium 专栏
- [PyTorch 2.0 Release](https://pytorch.org/blog/pytorch-2.0-release/) - PyTorch 2.0 发布公告

## 深度技术解析

### Edward Yang 的 PyTorch Internals 系列

Edward Yang（PyTorch 核心开发者）撰写的深度技术博客：

- [PyTorch Internals Part 1](http://blog.ezyang.com/2019/05/pytorch-internals/) - Tensor 和 Storage
- [Let's talk about the PyTorch dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) - 算子分发系统详解
- [PyTorch's tracing vs scripting](http://blog.ezyang.com/2019/05/pytorch-jit-tracing-vs-scripting/) - JIT 追踪与脚本化
- [How to call a C++ function from Python](http://blog.ezyang.com/2019/05/pytorch-internals-how-to-call-a-cpp-function-from-python/) - Python/C++ 互操作

### PyTorch 2.0 编译器栈

- [TorchDynamo Design Doc](https://github.com/pytorch/torchdynamo/blob/main/DESIGN.md) - TorchDynamo 设计文档
- [TorchInductor Architecture](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler/683) - TorchInductor 架构说明
- [Getting Started with FX](https://pytorch.org/docs/stable/fx.html) - FX 入门指南
- [Torch.compile FAQ](https://pytorch.org/docs/stable/torch.compiler_faq.html) - torch.compile 常见问题

### Autograd 与微分

- [Automatic differentiation in PyTorch](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) - Autograd 教程
- [PyTorch Autograd Explained](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95) - Autograd 原理解析
- [Extending PyTorch Autograd](https://pytorch.org/tutorials/advanced/cpp_autograd.html) - 扩展 Autograd

### 分布式训练

- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - DDP 官方教程
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html) - FSDP 官方文档
- [Getting Started with Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html) - 分布式训练入门

### 性能优化

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) - 性能调优指南
- [CUDA Best Practices](https://pytorch.org/docs/stable/notes/cuda.html) - CUDA 最佳实践
- [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html) - 混合精度文档
- [Memory Management and CUDA Caching Allocator](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) - CUDA 内存管理

## 社区资源

### 视频课程

- [Stanford CS231n](http://cs231n.stanford.edu/) - 计算机视觉与深度学习（使用 PyTorch）
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - 实用深度学习课程
- [PyTorch Official YouTube Channel](https://www.youtube.com/c/PyTorchLightning) - PyTorch 官方视频

### 书籍

- **《Deep Learning with PyTorch》** by Eli Stevens, Luca Antiga, Thomas Viehmann
  - [在线版本](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
- **《Programming PyTorch for Deep Learning》** by Ian Pointer
- **《PyTorch Recipes》** by Pradeepta Mishra

### 论文

#### PyTorch 核心论文

- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703) (NeurIPS 2019)
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) (VLDB 2020)

#### 编译器栈相关

- [TorchDynamo: Dynamic Tracing for PyTorch](https://openreview.net/forum?id=UwyOKTjYqjV) (MLSys 2023)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

#### Autograd 与微分

- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)

## 工具与生态

### 调试与性能分析

- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - 性能分析工具
- [TensorBoard with PyTorch](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) - TensorBoard 集成
- [torch.autograd.profiler](https://pytorch.org/docs/stable/autograd.html#profiler) - Autograd 性能分析器

### 模型部署

- [TorchScript](https://pytorch.org/docs/stable/jit.html) - 模型序列化和部署
- [ONNX](https://onnx.ai/) - 开放神经网络交换格式
- [TorchServe](https://pytorch.org/serve/) - PyTorch 模型服务化工具

### 扩展库

- [torchvision](https://pytorch.org/vision/stable/index.html) - 计算机视觉工具包
- [torchaudio](https://pytorch.org/audio/stable/index.html) - 音频处理工具包
- [torchtext](https://pytorch.org/text/stable/index.html) - 自然语言处理工具包
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - 高级训练框架
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Transformer 模型库

## 社区与交流

### 论坛与问答

- [PyTorch Forums](https://discuss.pytorch.org/) - 官方论坛
- [Stack Overflow - PyTorch Tag](https://stackoverflow.com/questions/tagged/pytorch) - Stack Overflow 问答
- [Reddit r/pytorch](https://www.reddit.com/r/pytorch/) - Reddit 社区

### 会议与活动

- [PyTorch Conference](https://pytorch.org/ecosystem/ptc/) - PyTorch 年度会议
- [PyTorch DevCon](https://pytorch.org/ecosystem/) - 开发者大会

### 社交媒体

- [PyTorch Twitter](https://twitter.com/PyTorch) - 官方 Twitter
- [PyTorch LinkedIn](https://www.linkedin.com/company/pytorch/) - 官方 LinkedIn

## 竞品与替代方案

### 其他深度学习框架

- [TensorFlow](https://www.tensorflow.org/) - Google 开发的深度学习框架
- [JAX](https://github.com/google/jax) - Google 的可微分编程框架
- [MXNet](https://mxnet.apache.org/) - Apache 基金会的深度学习框架
- [PaddlePaddle](https://www.paddlepaddle.org.cn/) - 百度开发的深度学习平台

### PyTorch 生态系统对比

| 特性 | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| 动态图 | ✅ 原生支持 | ✅ Eager Execution | ✅ 原生支持 |
| 静态图编译 | ✅ torch.compile | ✅ @tf.function | ✅ jit |
| 分布式训练 | ✅ DDP/FSDP | ✅ MirroredStrategy | ✅ pmap |
| 移动端部署 | ✅ PyTorch Mobile | ✅ TensorFlow Lite | ❌ 较弱 |
| 生态成熟度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 本文档贡献者

本 PyTorch 源码分析文档由以下贡献者整理和维护：

- 主要维护者：[SuperMarioYL](https://github.com/SuperMarioYL)
- 文档仓库：[inference-cookbook](https://github.com/SuperMarioYL/inference-cookbook)

### 参与贡献

欢迎通过以下方式参与贡献：

1. **GitHub Issues**：报告错误或提出改进建议
   - [提交 Issue](https://github.com/SuperMarioYL/inference-cookbook/issues)

2. **Pull Requests**：提交文档修正或新增内容
   - [提交 PR](https://github.com/SuperMarioYL/inference-cookbook/pulls)

3. **讨论与反馈**：参与技术讨论
   - [GitHub Discussions](https://github.com/SuperMarioYL/inference-cookbook/discussions)

## 许可证

本文档内容基于 [MIT License](https://opensource.org/licenses/MIT) 开源。

---

**最后更新**：2025-01-13

**文档版本**：v2.0（章节重组版）
