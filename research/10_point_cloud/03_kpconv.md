# KPConv: Flexible and Deformable Convolution for Point Clouds

**论文**: Thomas et al., ICCV 2019
**官方代码**: https://github.com/HuguesTHOMAS/KPConv-PyTorch

## 核心问题

图像卷积有规整的网格结构，点云没有。KPConv 在 3D 连续空间定义卷积核，通过"核点"（kernel points）将卷积操作从规整网格推广到任意点集。

## 关键设计：核点卷积

在 3D 空间放置 K 个核点 {p̃ₖ}，每个核点有一个权重矩阵 Wₖ。对于输入点 x，卷积为：

```
(KPConv * F)(x) = Σ_{xᵢ ∈ N(x)} Σ_{k=1}ᴷ h(xᵢ - x, p̃ₖ) Wₖ fᵢ
```

其中 h 是核点与邻域点之间的相关性函数（距离核函数）：

```
h(y, p̃ₖ) = max(0, 1 - ||y - p̃ₖ||/σ)
```

### 核点布局

- **Rigid KPConv**: 核点固定在球面上（如正多面体顶点），各向同性
- **Deformable KPConv**: 核点位置可学习偏移 Δ(p̃ₖ)，适应局部几何

偏移学习：

```
Δ(p̃ₖ) = Σ_{xᵢ ∈ N(x)} h(xᵢ - x, p̃ₖ) (xᵢ - p̃ₖ) / Σ h(...)
```

即核点向局部点密度高的方向移动。

## 网络结构：KP-FCNN

```
编码器: 5层 KPConv + 4次下采样（grid subsampling）
解码器: 4层最近邻上采样 + skip connections
```

Grid Subsampling：体素化下采样（控制输入点数量），比 FPS 快。

## 关键 insight

- 核点相当于在 3D 空间学习"卷积核的形状"
- Deformable 版本给每个位置学习适配的核点偏移，类似 Deformable CNN
- 在分割任务上显著超越 PointNet++

## 实验

- S3DIS 语义分割: OA 86.4%, mIoU 67.1%
- Semantic3D: 超越当时所有方法
- 速度比 PointNet++ 更快（Grid Subsampling vs FPS）
