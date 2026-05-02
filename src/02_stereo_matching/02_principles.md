# B.2 原理解析

> **目标**：理解三条技术路线的核心设计——基于 cost volume 的迭代细化（RAFT-Stereo/CREStereo）、多视图 MVS（MVSNet）、端到端点图回归（DUSt3R）。知道它们的关键架构差异和演进逻辑。

## Cost Volume 范式

双目匹配的核心问题是：给定左图的一个像素 $(u,v)$，在右图的对极线上找到它的对应点 $(u-d, v)$，输出视差 $d$。

在深度学习时代之前，这个问题的标准解法是 SGM（Semi-Global Matching, Hirschmuller 2011）：先计算每个像素在每个候选视差下的匹配代价（用 Census transform 或互信息），然后通过全局能量函数平滑视差图。SGM 是工程上极其成功的算法——在 GPU 上可以实时运行，精度在大多数场景下足够。但它在弱纹理和遮挡区域表现差，因为它只能利用局部的光度信息做匹配。

2015 年，Zbontar & LeCun 的 MC-CNN 首次用 CNN 学匹配代价——从图像 patch 中提取深度特征，用特征的相似度作为匹配代价。精度大幅超过手工特征，验证了"学习的特征比手工特征好"这个关键假设。

### 3D Cost Volume

MC-CNN 虽然特征好了，但匹配本身还是逐像素的。后续工作（PSMNet, GCNet）提出了一个更结构化的方法：**3D cost volume**。

给定左右图的深度特征图 $F_L, F_R \in \mathbb{R}^{H \times W \times C}$，对每个候选视差 $d$，将 $F_R$ 平移 $d$ 列后与 $F_L$ 拼接（或做内积），得到一个 $H \times W \times D_{max} \times C$ 的 4D volume。然后用 3D 卷积在这个 volume 上做正则化，最后用 soft-argmin 回归每个像素的视差：

$$\hat{d} = \sum_{d=0}^{D_{max}} d \cdot \text{softmax}(-c_d)$$

其中 $c_d$ 是 cost volume 中视差层 $d$ 的匹配代价。Soft-argmin 是可微的，所以整个网络（特征提取 + cost volume + 3D CNN + soft-argmin）可以端到端训练。

3D cost volume 的问题是**计算量极大**——3D 卷积的 FLOPs 随视差范围 $D_{max}$ 线性增长，显存占用也随 $D_{max}$ 增长。这限制了它在高分辨率图像和大视差范围场景中的应用。

## 迭代细化路线：RAFT-Stereo (3DV 2021, Best Paper)

Lipson 等人（Princeton）在 2021 年提出的 RAFT-Stereo，借鉴了他们之前在光流估计上的 RAFT 架构，用**迭代细化**替代了一次性的 3D cost volume。

### 架构

RAFT-Stereo 有三个核心模块：

**特征编码器**：两个独立的 CNN 编码器。Feature Encoder 从左右图提取匹配用的特征（$H/4$ 分辨率, 256 通道），Context Encoder 同样架构但只用左图（初始化 GRU 隐藏状态）。Feature Encoder 用 instance norm，Context Encoder 用 batch norm。

**3D 相关金字塔**：不同于 RAFT 的 4D all-pairs correlation，RAFT-Stereo 构建的是 **3D 相关体积**（$H \times W \times W$）。它利用矫正后双目系统的对极约束——对应点一定在同一行上——只在水平方向计算点积。从全分辨率开始，沿视差维度做 4 级 1D 平均池化，构成金字塔。这比 4D volume 节省了大量计算和显存。

**多级 GRU 更新算子**（核心创新）：RAFT-Stereo 使用三个 convolutional GRU，分别工作在 $1/8$、$1/16$、$1/32$ 输入分辨率。GRU 之间**交叉连接**——每个 GRU 接收相邻分辨率 GRU 的隐藏状态（上采样或下采样后）。只有最高分辨率的 GRU 做相关查找（correlation lookup）并输出视差更新。

多级设计的动机是**感受野增长速度**。单级 GRU 需要在很多次迭代后才能"看到"大范围上下文，而低分辨率 GRU 在少量迭代内就能覆盖很大的有效感受野（因为低分辨率下每个像素代表更大的物理区域），然后通过交叉连接把全局上下文传递给高分辨率 GRU。

**迭代更新**：初始视差 $d_0 = 0$。每步：视差索引相关金字塔 → 相关特征 + 当前视差 + 上下文特征 → GRU 更新隐藏状态 → 最高分辨率 GRU 输出视差残差 $\Delta d$ → 叠加。所有迭代的预测序列 $\{d_1, ..., d_N\}$ 用指数加权 L1 loss 监督。

**Slow-Fast 变体**：高分辨率 GRU 的 FLOPs 约是低分辨率的 4 倍。Slow-Fast 策略让低分辨率 GRU 更新更多次（1/32 GRU: 30 次, 1/16: 20 次），高分辨率 GRU 更新更少次（1/8: 10 次）——推理速度从 0.132s 降到 0.05s（52%），精度几乎不降。

**结果**：Middlebury 基准排名第一（bad-2px 4.74%），ETH3D 所有已发表工作中排名第一（bad-1px 2.44）。

## 自适应相关路线：CREStereo (CVPR 2022, Oral)

RAFT-Stereo 假设图像对已经过完美矫正——对应点严格在同一行上。但真实场景中矫正不完美、镜头畸变残差、或者 rolling shutter 效应，都会让对应点偏离对极线。CREStereo（Megvii, CVPR 2022 Oral）的动机就是**处理不完美矫正和困难纹理的真实场景。**

### 自适应组相关层（AGCL）

CREStereo 的核心模块。与 RAFT-Stereo 的 1D 对极线搜索不同，AGCL 做 **2D-1D 交替局部搜索**：先在 2D 邻域内搜索（处理垂直偏移），再做 1D 水平搜索。搜索窗口是**可变形的**——偏移量由可学习的参数决定，让网络自适应地"找到"正确的搜索区域。

第一级级联中，AGCL 集成了 LoFTR 风格的局部特征注意力（self/cross-attention + 位置编码），增强在弱纹理区域的匹配能力。特征按组划分后计算组内点积（group-wise correlation，来自 GwcNet），每组独立计算匹配代价。

### 级联循环网络

CREStereo 在 3 个分辨率层级上（$1/16, 1/8, 1/4$）各部署一个循环更新模块（RUM），每个 RUM 包含 GRU + AGCL。所有 RUM **共享权重**——不同分辨率用同样的匹配逻辑。推理时，处理高分辨率图像通过图像金字塔 + 跳跃连接，不需要额外微调。

### 合成数据集

CREStereo 专门为困难场景构建了一个 ~400GB 的 Blender 合成数据集，强调：薄结构、无纹理/重复纹理区域、复杂光照、金属反射、透明物体。这个数据集是其真实世界泛化能力的关键来源——在这些困难 case 上训练过的模型，在真实反射和弱纹理场景中表现显著更好。

**结果**：Middlebury 2014 排名第一（bad-2.0 = 3.71 vs RAFT-Stereo 4.74），ETH3D 排名第一（bad-1.0 = 0.98，比此前最好方法提升 59.84%）。

## 多视图 MVS：MVSNet (ECCV 2018)

双目做的是"两张图恢复深度"。多视图立体匹配（Multi-View Stereo, MVS）把场景扩展到 N 张任意视角的图像——"从几十张照片重建整个场景的稠密 3D 模型。"

Yao 等人（HKUST）的 MVSNet 是学习型 MVS 的奠基工作。

### 可微分单应变换

核心思想：**plane sweep**。在参考相机前方按深度采样一系列平行平面（plane sweep volumes）。对每个候选深度，用单应矩阵 $H(d)$ 将源视角的特征图 warp 到参考视角：

$$H_i(d) = K_i \left( R_i R_0^{-1} + \frac{(t_i - R_i R_0^{-1} t_0) n^T}{d} \right) K_0^{-1}$$

其中 $K_0, K_i$ 是参考和源相机的内参，$R, t$ 是相对位姿，$n$ 是参考相机的主轴方向，$d$ 是候选深度。这个公式可微——梯度可以从 loss 反向传播到特征提取网络，**使整个 pipeline 端到端可训练**。

### 基于方差的代价度量

所有源视角的特征 warp 到参考视角后，如何融合？MVSNet 用 **方差**（而不是拼接或求和）：

$$C = \frac{1}{N} \sum_{i=1}^{N} (V_i - \bar{V})^2$$

方差度量对任意 $N$ 个视角都适用（不像拼接要求固定输入数量），而且方差自然地惩罚不一致的特征——如果某个视角的特征因遮挡而异常，它的贡献会在取平均时被稀释。

### 3D CNN 正则化 + Soft-Argmin

代价体积 $C$ 用多尺度 3D CNN 做正则化（去噪、填洞），然后 softmax 沿深度维度输出概率体积。最终深度由 soft-argmin 回归：

$$\hat{D} = \sum_{d} d \cdot P(d)$$

还有一个轻量级 refinement 网络用参考图像改善物体边界的深度。

**结果**：DTU 数据集 0.527mm 完整度误差，Tanks and Temples 排名第一（无需微调），验证了"可微单应 warp + 方差融合 + 3D CNN"这个范式。

## 端到端点图路线：DUSt3R (CVPR 2024)

DUSt3R（NAVER LABS, CVPR 2024）是对传统 MVS 管线的一次根本性重新思考。它不再把重建拆成"特征提取 → 匹配 → 三角测量"三步，而是**直接从图像对回归 3D 点图**。

### 核心设计

输入：两张没有标定的任意图像。输出：每张图对应的 3D 点图（每个像素一个 $(X,Y,Z)$，在统一的参考系中）。**不需要相机标定、不需要位姿先验、不需要对极约束。**

架构是标准的 Transformer encoder + decoder。训练目标是直接监督点图的 3D 坐标。

### 为什么重要

传统 MVS 管线的每个步骤（特征匹配、位姿估计、三角测量）都可能积累误差，且很难联合优化。DUSt3R 把这些全部"压"进一个端到端模型中——让网络自己去学"怎么从两个视角的图像中推断 3D 结构"。

对于多视图场景，DUSt3R 通过全局对齐策略把多个 pairwise 点图统一到同一个坐标系中——本质上是用可微的对齐替代了传统 SfM 中的 BA 优化。

MV-DUSt3R+ 把这个思路扩展到多视图，在稀疏视角下约 2 秒完成重建。

> **DUSt3R 和传统 MVS 的关系，有点像端到端自动驾驶和传统"感知→规划→控制"管线的关系：一个路线试图用学习替代全部手工模块，另一个路线在每个模块上精雕细琢。目前两条路线各有优劣——DUSt3R 在泛化性和便捷性上碾压，但传统 MVS 在已知场景下的精度天花板可能更高。**

## 路线对比

```mermaid
mindmap
  root((双目与MVS路线))
    Cost Volume范式
      3D/4D cost volume
      3D CNN正则化
      Soft-argmin回归
    RAFT-Stereo
      多级GRU迭代
      3D相关金字塔
      1D对极线搜索
    CREStereo
      自适应组相关
      2D-1D交替搜索
      共享权重级联
    MVSNet
      可微分单应变换
      方差代价融合
      多尺度3D CNN
    DUSt3R
      端到端点图回归
      无需标定
      Transformer架构
