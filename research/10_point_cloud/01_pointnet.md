# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

**论文**: Qi et al., CVPR 2017
**官方代码**: https://github.com/charlesq34/pointnet

## 核心问题

点云是无序的点集合。传统方法将点云转为体素网格或投影到多视图，丢失信息或计算昂贵。PointNet 是第一个直接在原始点云上做深度学习的网络。

## 关键设计：对称函数保证置换不变性

点云是集合 {x₁, x₂, ..., xₙ}，排列改变不应改变网络输出。PointNet 的方案：

```
f({x₁, ..., xₙ}) ≈ g(h(x₁), ..., h(xₙ))
```

其中 h 是共享 MLP，g 是对称函数（max pooling）。Max pooling 是置换不变的，因此整个网络对任意排列输出相同。

数学上：所有连续置换不变函数可以表示为 γ(MAX{h(xᵢ)}) 的形式（Universal Approximation Theorem for point sets）。

## 网络架构

```
输入点云 (n × 3)
  → 共享 MLP [64, 64] 
  → 共享 MLP [64, 128, 1024] 
  → Max Pooling (全局特征 1024-d)
  → 分类：MLP [512, 256, k]
  → 分割：拼接全局特征 + 每点局部特征 → MLP
```

## T-Net：学习输入变换

- 第一个 T-Net 学 3×3 变换矩阵（空间对齐）
- 第二个 T-Net 学 64×64 特征变换
- 加正则化项使变换矩阵接近正交

## 理论贡献

**Theorem 1**: PointNet 可以近似任意连续集合函数，给定足够的隐藏层宽度。

**分析**: 关键点集（critical points）——max pooling 选出对输出贡献最大的点，形成点云的"骨架"（skeleton）。网络的鲁棒性来自只有少数 critical points 决定输出。

## 实验

- ModelNet40 分类：89.2% acc
- ShapeNet part segmentation: mIoU 83.7%
- 对数据损坏（点丢失、异常点）的鲁棒性远超体素方法

## 局限性

- 只捕获全局特征，无法建模局部结构——这催生了 PointNet++
- Max pooling 丢失了大部分几何信息
