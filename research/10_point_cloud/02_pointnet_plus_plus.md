# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

**论文**: Qi et al., NeurIPS 2017
**官方代码**: https://github.com/charlesq34/pointnet2

## 核心问题

PointNet 只做了一次全局 max pooling，无法捕获局部结构。但点云的局部结构恰恰是理解 3D 几何的关键——角点、边、曲面都需要局部邻域信息。

## 关键设计：层级 Set Abstraction

Set Abstraction = Sampling + Grouping + PointNet

```
层级 1: 输入 N×(d+C) → 采样 N₁ 个中心点 → 以半径 r₁ 分组 → PointNet 提取局部特征 → 输出 N₁×(d+C₁)
层级 2: 输出 N₂×(d+C₂)
...
```

### 采样（Farthest Point Sampling, FPS）

- 从点集中选最远点子集
- 保证覆盖整个点云
- 复杂度 O(n²)，但在 GPU 上可优化

### 分组（Ball Query / kNN）

- Ball Query: 以半径 r 内的所有点为一组
- 多尺度分组 (MSG): 多个半径 {r₁, r₂, r₃}，多尺度特征拼接
- 多分辨率分组 (MRG): 不同抽象级别的特征拼接

### PointNet 层

对每个分组的局部点集应用 mini-PointNet，提取局部特征向量。

## 密度自适应

点云密度变化（近处密、远处疏）是核心挑战：
- **MSG**: 不同半径的 ball query，学习跨尺度特征
- **MRG**: 高分辨率区域用低层特征，低分辨率区域用高层特征

## 分割任务：特征传播

分类只需全局特征；分割需要每点特征。用反向插值将特征从子采样点传播回原分辨率：

```
f(x) = Σ wᵢ(x) fᵢ / Σ wᵢ(x)
wᵢ(x) = 1 / d(x, xᵢ)ᵖ
```

即基于距离的反距离加权插值，再通过 skip connections 拼接编码器对应层的特征。

## 实验

- ModelNet40 分类：91.9% (PointNet: 89.2%)
- ScanNet 语义分割：显著超越 PointNet
- 在非均匀采样下的鲁棒性大幅提升

## 关键 insight

"PointNet++ 之于点云，就像 CNN 之于图像"——层级化局部特征提取是普适的范式。
