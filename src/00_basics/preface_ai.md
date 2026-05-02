# 插曲：你需要知道的一点 AI 前置知识

> **阅读时间**：25 分钟 · **位置**：基础篇之后，主题模块之前

后续模块 A/B/C 里你会频繁遇到 CNN、Vision Transformer、扩散模型、损失函数这些概念。本节用最短的篇幅给你一个"能看懂论文"的 AI 基础。每个话题只讲最关键的一两句话，并给出进一步学习的推荐。

**本节不替代系统学习。** 如果你需要深入理解某个网络结构或训练方法，文末有推荐资源。

## 神经网络的基本构成

一个神经网络就是一堆可微分的函数叠在一起。给定输入 $x$，经过若干层变换，得到输出 $\hat{y}$。然后用一个**损失函数** $\mathcal{L}(\hat{y}, y)$ 比较预测值和真实值的差异，通过反向传播（链式法则求梯度）更新每一层的参数。

核心概念：

| 概念 | 一句话 | 对应到深度估计里 |
|------|--------|----------------|
| **卷积层（Conv）** | 用一个小窗口在图像上滑动，每次只看窗口内的像素 | CNN 提取边缘、纹理等局部特征 |
| **池化层（Pooling）** | 把相邻像素合并成一个（取最大或平均），让特征图缩小 | 减少计算量，增大感受野 |
| **全连接层（FC）** | 每个输入连接到每个输出，最"笨"但最灵活 | 最后的输出头 |
| **激活函数** | 给网络引入非线性。ReLU（$f(x)=\max(0,x)$）最常见 | 没有它，100 层网络 = 1 层线性变换 |
| **Batch Normalization** | 把每一层输出的均值和方差拉到一个固定范围 | 训练更快、更稳定 |

这些概念在 2015 年以前是计算机视觉的全部。

## CNN 做密集预测

基础篇讲的是相机模型和几何。模块 A/B/C 要用的深度学习方法，最早都来自 CNN。

CNN 最初是为图像分类设计的（输入一张图，输出一个标签）。但深度估计需要输出**和输入同样大小的密集预测**（每个像素一个值）。为此需要：

- **上采样**：把缩小的特征图放大回原图尺寸。最近邻/双线性插值最简单，转置卷积（transposed convolution）可学习。
- **跳跃连接（Skip Connection）**：把浅层的高分辨率特征和深层的语义特征拼接起来。UNet 是最经典的例子——形状像 U，编码器逐层缩小，解码器逐层放大，同级之间有横向连接。
- **空洞卷积（Dilated Conv）**：不增加计算量的前提下增大感受野——隔几个像素取一次。

> **推荐**：Stanford CS231n (Fei-Fei Li) 的公开课对 CNN 体系有完整讲解。Stanford CS231n 笔记搜索 "CS231n Convolutional Neural Networks"。

## Vision Transformer (ViT)

2021 年之后，CNN 的地位被 Vision Transformer（ViT）逐渐取代。ViT 把图像切成小块（patch），每一块当成 NLP 里的一个"词"，然后用 Transformer 的自注意力机制处理。

**自注意力（Self-Attention）的核心公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

三个矩阵 $Q$（Query）、$K$（Key）、$V$（Value）都来自同一个输入。$QK^T$ 计算每对 patch 之间的"相关度"，$\text{softmax}$ 把它们变成权重，最后乘 $V$ 得到加权输出。物理含义：**每个 patch 去看看其他所有 patch，决定应该从它们那里"借鉴"多少信息。**

**为什么 ViT 比 CNN 好（对密集预测任务）**：
- CNN 只看局部（conv kernel 大小受限），ViT 全局自注意力——远处的像素也能直接交互
- ViT 的特征在整个图像上保持一致（CNN 有"平移不变性"悖论——太局部了就不变，太大了又丢了位置）

**ViT 的一个问题**：自注意力的计算量随 patch 数量呈 $O(n^2)$ 增长。所以在 3D 视觉中，多数模型用 ViT 做编码器（提取特征），然后用轻量的 CNN 做解码器（上采样到全分辨率）。

> **推荐**：Jay Alammar 的 "The Illustrated Transformer" 是目前最好的 Transformer 图文教程。搜索即可找到中英文版本。

## DINOv2

DINOv2（Meta AI, 2023）是当前 3D 视觉中最广泛使用的**视觉基础模型骨干**。Depth Anything V1/V2、MoGe 等都用 DINOv2 作为编码器。

DINOv2 的训练方式是**自监督学习**：在不使用任何人工标注的情况下，在 142M 张图像上训练 ViT。方法叫 DINO（self-**DI**stillation with **NO** labels）——teacher 模型和 student 模型看同一张图的不同裁剪，让 student 去匹配 teacher 的输出。两个模型同步更新。

**为什么 DINOv2 对深度估计特别好？** 它的自监督目标迫使模型理解物体的形状、遮挡关系、空间结构——因为要匹配不同视角下的同一物体，模型必须隐式地学会这些几何信息。论文实验显示，DINOv2 特征在 frozen evaluation 下，仅加一个线性探针就能做到不错的深度估计——说明几何信息已经被自监督目标"锁"在了特征里。

> **推荐**：DINOv2 论文 (Oquab et al., 2023) + Meta 官方博客。DeepWiki 上有结构化的论文笔记。

## 扩散模型 (Diffusion Models)

扩散模型是 2022-2024 年图像生成领域的统治性方法（Stable Diffusion, DALL-E 3, Midjourney）。Marigold 把它们用在了深度估计上。

**核心直觉**：
- **前向过程**：给一张干净的图像逐步加高斯噪声，加到第 $T$ 步时变成纯噪声。
- **反向过程**：训练一个网络从噪声中恢复出干净图像——学会"去噪"。
- 去噪网络通常用 UNet 架构（编码器-解码器 + 跳跃连接）。

**为什么扩散模型能用来做深度估计？** 因为 Stable Diffusion 在数十亿张图像上预训练后，其 UNet 内部已经学到了极其丰富的视觉先验——物体该长什么样、场景怎么布局、光从哪来。Marigold 的做法是微调 UNet 的一小部分，让它的输入从"加噪的 RGB"变成"加噪的 RGB + 噪声深度"，输出从"干净 RGB"变成"干净深度"。

> **推荐**：Lilian Weng 的博客 "What are Diffusion Models?" 是目前最好的扩散模型入门教程。Calvin Luo 的 "Understanding Diffusion Models" 也不错。

## 损失函数速查

后续模块会反复出现这些损失函数：

| 损失函数 | 公式 | 用在哪 | 特点 |
|---------|------|--------|------|
| **L1 Loss (MAE)** | $\frac{1}{N}\sum \vert \hat{y}_i - y_i \vert$ | metric 深度监督 | 对 outlier 不敏感，边缘更锐利 |
| **L2 Loss (MSE)** | $\frac{1}{N}\sum (\hat{y}_i - y_i)^2$ | 通用回归 | 对 outlier 敏感，过度光滑 |
| **SSI Loss** | 先做 scale-shift 对齐再算 L1/L2 | 单目深度估计 | 核心创新——让不同尺度的数据可以混合训练 |
| **BerHu Loss** | L1（小误差） + L2（大误差） | Laina et al. 深度估计 | L1 和 L2 的折中 |
| **Gradient Loss** | $\frac{1}{N}\sum \|\nabla\hat{y}_i - \nabla y_i\|_1$ | Depth Anything V2 | 强制模型学习锐利边缘 |
| **Cross-Entropy** | $-\sum y_i \log \hat{y}_i$ | 分类任务 | 深度估计不用 |
| **Dice Loss** | $1 - \frac{2\vert A\cap B\vert}{\vert A\vert+\vert B\vert}$ | 分割任务 | 处理类别不平衡 |

## 训练范式

| 范式 | 做法 | 代表 |
|------|------|------|
| **监督学习（Supervised）** | 用带 GT 深度标签的数据训练 | 早期所有方法 |
| **自监督学习（Self-Supervised）** | 用双目对或多帧视频的一致性约束训练（不需要深度 GT） | Monodepth2, FeatDepth |
| **半监督（Semi-Supervised）** | 少量标签 + 大量无标签数据 | — |
| **蒸馏（Distillation）** | 大 teacher 模型给小 student 模型打伪标签，student 用伪标签训练 | Depth Anything V2 |
| **微调（Fine-tuning）** | 在预训练基础模型上，用少量目标域数据继续训练 | Depth Anything metric 版 |

## 评估指标

读论文时会看到这些指标来衡量深度估计精度。全称不重要，理解"值小就好"的方向即可：

| 指标 | 方向 | 含义 |
|------|------|------|
| **AbsRel** (Absolute Relative Error) | ↓↓ | $\frac{1}{N}\sum \frac{|d_i - d_i^*|}{d_i^*}$ —— 平均相对误差。最常用 |
| **RMSE** (Root Mean Squared Error) | ↓↓ | $\sqrt{\frac{1}{N}\sum (d_i-d_i^*)^2}$ —— 对大误差惩罚重 |
| **$\delta < 1.25$** | ↑↑ | $\frac{1}{N} \sum \left[\max(\frac{d_i}{d_i^*},\frac{d_i^*}{d_i})<1.25\right]$ —— "预测值和真实值的比在 1.25 倍以内的像素比例"。越高越好 |
| **log10** | ↓↓ | $\frac{1}{N}\sum |\log_{10}d_i - \log_{10}d_i^*|$ —— 对数空间误差 |

> 训练/测试时用的是**单帧和 GT 深度**的比较，而不是多帧序列的重建质量。所以模型可能在 AbsRel 上很好，但把多帧深度拼成 3D 场景时出现几何不一致。这是当前 depth foundation model 的一个已知缺陷。

## 进一步学习

- **深度学习体系**：Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016). 免费在线版：deeplearningbook.org
- **CNN + CV 入门**：Stanford CS231n (Fei-Fei Li et al.). 讲义和作业在线公开
- **Transformer 图解**：Jay Alammar, "The Illustrated Transformer"
- **扩散模型详解**：Lilian Weng, "What are Diffusion Models?"
- **深度估计综述**：Xu et al., *Towards Depth Foundation Models* (CVM 2026) — 就是 research 文件夹里那篇
