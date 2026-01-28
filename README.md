# vLLM 原理深度解析

> 面向深度学习初学者的 vLLM 技术文档

**在线阅读**: [https://vllmbook.lei6393.com](https://vllmbook.lei6393.com)

## 简介

本文档系列旨在帮助深度学习初学者深入理解 vLLM —— 一个高性能的大语言模型（LLM）推理和服务框架。

## 本地开发

### 前置要求

- [Hugo](https://gohugo.io/installation/) (Extended 版本 v0.146.0+)

### 启动本地服务器

```bash
# 克隆仓库（包含主题）
git clone --recursive https://github.com/SuperMarioYL/vllm-book.git
cd vllm-book

# 启动开发服务器
hugo server --buildDrafts

# 访问 http://localhost:1313
```

### VSCode 调试

项目已配置 VSCode 调试支持：
- 按 `F5` 启动 Hugo Server
- 或使用 `Cmd+Shift+B` 运行构建任务

### 构建

```bash
hugo --gc --minify
```

构建产物在 `public/` 目录下。

## 目录结构

```
content/docs/
├── 01-introduction/      # 入门篇
├── 02-dl-fundamentals/   # 深度学习基础
├── 03-core-modules/      # 核心模块详解
├── 04-code-walkthrough/  # 代码链路分析
├── 05-advanced-topics/   # 进阶主题
└── appendix/             # 附录
```

## 技术栈

- [Hugo](https://gohugo.io/) - 静态网站生成器
- [hugo-book](https://github.com/alex-shpak/hugo-book) - 文档主题
- [GitHub Pages](https://pages.github.com/) - 托管平台
- [GitHub Actions](https://github.com/features/actions) - CI/CD

## 许可

MIT License
