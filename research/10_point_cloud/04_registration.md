# 点云配准：从 ICP 到 TEASER++

## ICP: Iterative Closest Point (Besl & McKay, 1992)

**核心思想**: 迭代交替 (1) 找最近点对应 (2) 解刚体变换

```
while not converged:
    C = find_correspondences(P, Q)  # 最近邻
    R, t = argmin Σ ||Rpᵢ + t - q_c(i)||²  # SVD / 四元数
```

**局限性**: 
- 需要好的初始位姿（局部收敛）
- 对异常点/噪声敏感
- 大面积重叠假设

**变体**:
- Point-to-plane ICP: 更快收敛
- GICP (Generalized ICP): 概率框架

## FGR: Fast Global Registration (Zhou et al., ECCV 2016)

**核心创新**: 不使用最近邻对应，而是优化 FPFH 特征匹配。

两阶段：
1. 用 FPFH 特征做双向匹配（mutual nearest neighbors）
2. 鲁棒优化：用 Geman-McClure 鲁棒损失 + 交替优化（线过程）

目标函数：

```
E(R, t) = Σ ρ(||Rpᵢ + t - qᵢ||)
ρ(x) = μx²/(μ + x²)  # Geman-McClure
```

用线过程（line process）引入隐变量 lᵢ：

```
E(R, t, {lᵢ}) = Σ lᵢ||Rpᵢ + t - qᵢ||² + Σ Ψ(lᵢ)
```

交替优化：(R, t) 和 {lᵢ}，每步有闭式解。

**速度**: 几十毫秒到几百毫秒，比 RANSAC 快 10-100 倍

## TEASER++: Certifiably Robust Registration (Yang et al., 2021)

**核心创新**: 可认证的鲁棒配准——给定外点比例，保证找到全局最优解。

两步解耦：
1. **尺度 + 旋转估计**: Truncated Least Squares (TLS) + Graduated Non-Convexity (GNC)
2. **平移估计**: 自适应投票

### Truncated Least Squares (TLS)

```
min Σ min(dᵢ² / c̄², 1)  # 截断损失，外点贡献不超过 1
```

GNC 将 TLS 松弛为一系列加权最小二乘子问题：

```
λₖ 从 1 → λ_max
min Σ wᵢ(λₖ) dᵢ²  # 加权最小二乘
权重 wᵢ 从 1 逐渐趋于 0（对外点）
```

### 旋转估计

将 GNC-TLS 用于旋转搜索，使用**与旋转无关的测量**（TIM: Translation Invariant Measurements）：

测量 rᵢⱼ = aᵢ - aⱼ（两点的差），估计 R 使 Rrᵢⱼ ≈ r'ᵢⱼ

用 quaternion + GNC 高效求解。

### 平移估计

给定 R，平移 t 的 TLS 问题在 (x,y,z) 每个维度独立，可在 max clique 范围内自适应投票。

**鲁棒性**: 可容忍 90%+ 外点（相同比例下 RANSAC 需要 10⁶ 次迭代，TEASER 几秒内求解到全局最优）

## 深度配准：GeoTransformer (Qin et al., ECCV 2022)

- 用 KPConv/Transformer backbone 提取超点特征
- Geometric Self-Attention: 在超点间建图注意力
- Superpoint Matching: 粗匹配 → 精化
- 在 3DMatch 和 KITTI 上 SOTA
