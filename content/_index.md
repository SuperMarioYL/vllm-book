---
title: "vLLM Cookbook"
linkTitle: "vLLM Book"
---

{{< blocks/cover title="vLLM Cookbook" image_anchor="top" height="full" >}}
<a class="btn btn-lg btn-primary me-3 mb-4" href="{{< relref "/docs" >}}">
  开始学习 <i class="fas fa-arrow-alt-circle-right ms-2"></i>
</a>
<a class="btn btn-lg btn-secondary me-3 mb-4" href="https://github.com/SuperMarioYL/vllm-book">
  GitHub <i class="fab fa-github ms-2 "></i>
</a>
<p class="lead mt-5">面向深度学习初学者的 vLLM 技术文档</p>
{{< /blocks/cover >}}

{{% blocks/lead color="primary" %}}
本文档系列旨在帮助深度学习初学者深入理解 vLLM —— 一个高性能的大语言模型（LLM）推理和服务框架。

我们将从最基础的概念出发，逐步深入到核心算法和代码实现，让你不仅知其然，更知其所以然。
{{% /blocks/lead %}}

{{% blocks/section color="dark" type="row" %}}
{{% blocks/feature icon="fa-lightbulb" title="理解核心原理" %}}
深入讲解 PagedAttention 和连续批处理等核心创新技术
{{% /blocks/feature %}}

{{% blocks/feature icon="fab fa-github" title="源码级分析" url="https://github.com/vllm-project/vllm" %}}
跟踪代码执行路径，理解从请求到响应的完整链路
{{% /blocks/feature %}}

{{% blocks/feature icon="fa-graduation-cap" title="循序渐进" %}}
从深度学习基础到进阶主题，提供清晰的学习路径
{{% /blocks/feature %}}
{{% /blocks/section %}}

{{% blocks/section %}}
## 适用读者

- **深度学习初学者**：了解基本的 Python 编程，对机器学习有初步认识
- **LLM 应用开发者**：希望了解推理框架底层原理
- **系统工程师**：负责部署和优化 LLM 服务
- **研究人员**：研究 LLM 推理优化技术
{{% /blocks/section %}}
