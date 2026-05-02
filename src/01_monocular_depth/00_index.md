# 模块 A：单目深度估计

> **一句话**：给一张普通 RGB 照片，推断每个像素离相机有多远——不需要激光雷达，不需要第二台相机，只靠一张图。

## 为什么这个模块重要

单目深度估计是 3D 视觉中最"轻"的深度获取方式。不需要特殊硬件，不需要多台相机校准，不需要移动相机拍多张照片——只要一张图，模型就能输出每个像素的距离。

这件事在 2019 年之前基本不可行。单目深度估计本质上是一个 ill-posed 问题（无穷多三维场景可以投影出同一张二维图像），传统方法要么需要 motion（SfM），要么需要 stereo pair。

但深度学习改变了这个局面。从 CNN 到 Transformer 到扩散模型，从 ImageNet 预训练到大规模混合数据集训练（10M+ 图像），单目深度估计在 2023-2025 年经历了爆发式进步——某些基础模型已经可以在任意场景、任意相机下以厘米级精度输出 metric 深度。

## 本模块学什么

| 节 | 主题 | 核心收获 |
|----|------|---------|
| 01 直观理解 | 为什么单目深度是 ill-posed？人为什么能做到？relative vs metric | 建立问题直觉，理解边界 |
| 02 原理解析 | CNN/Transformer/Diffusion 三条路线，关键模型拆解 | 理解 MiDaS、DPT、Marigold、Depth Anything、MoGe 的设计差异 |
| 03 部署实战 | 用 Depth Anything V2 和 MoGe 跑照片，对比精度和速度 | 会选模型、会跑推理 |

## 前置知识

- 基础篇 01：相机模型（归一化坐标、内参 $K$、畸变）
- 基础篇 04：深度表示（深度图 vs 点云、视差公式 $Z = fb/d$）
- 基础篇 06：优化基础（损失函数概念）

不需要先学双目或多视图——单目深度估计的妙处正是它不需要那些几何约束。

## 关键论文速览

| 年份 | 工作 | 核心贡献 |
|------|------|---------|
| 2019 | Ranftl et al., *MiDaS* | 混合数据集训练，首次实现鲁棒 relative depth 估计 |
| 2021 | Ranftl et al., *DPT* | Vision Transformer 替代 CNN，精度大幅提升 |
| 2023 | Bhat et al., *ZoeDepth* | relative 预训练 + metric 微调的两步法 |
| 2024 | Yang et al., *Depth Anything v1/v2* | 大规模无标签数据蒸馏，foundation model 路线 |
| 2024 | Ke et al., *Marigold* | Stable Diffusion 先验嫁接至深度估计 |
| 2024 | Yin et al., *Metric3D v2* | 几何约束注入，解决跨相机 metric 精度 |
| 2024 | Wang et al., *MoGe* | 3D-aware 训练 + 全局几何一致性 |

> 先跳到 A.1 建立直觉。想理解技术路线差异看 A.2。需要跑代码直接看 A.3。
