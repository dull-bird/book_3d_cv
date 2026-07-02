# LRM 家族与 TripoSR：前馈式 3D 生成

## LRM: Large Reconstruction Model (Hong et al., ICLR 2024)

**核心问题**: 优化式方法（DreamFusion 等）太慢（几十分钟到几小时）。能否学一个神经网络，前馈一次就输出 3D？

**核心设计**: Transformer + Triplane 表示

### 架构

```
输入：单张 RGB 图像 (或稀疏多视图)
  → DINO 图像编码器 → 图像 token
  → 拼接相机参数 (Plücker ray embedding)
  → Transformer decoder（cross-attend to image tokens）
  → 输出 triplane token → 3 个平面 (3 × 64 × 64 × C)
  → 查询任意 3D 点的 triplane 特征 → 小 MLP → density + color
```

### Triplane 表示

Triplane = 三个正交平面对场景的分解（XY, XZ, YZ）。任意 3D 点投影到三个平面取特征，求和 → 小 MLP 解码为 density/RGB。

相比完整 3D 体素：O(n³) vs O(3n²)，大幅降维。
相比 NeRF bottleneck：triplane 是可卷积的、结构化的。

### 训练

- 数据: Objaverse（~800K 3D 模型），每个模型渲染 32 张多视图
- 损失: 渲染 RGB + 渲染 mask + LPIPS
- 单卡 A100 训练 3 天

### 效果

- 单张图片 → 3D mesh，前馈一次约 5-10 秒
- 比优化式方法快 100-1000 倍
- 但质量不如优化式（缺少 per-instance 精调）

## Instant3D (Li et al., ICLR 2024)

两步法：
1. 微调 SD 生成 4 个固定视角的视图
2. LRM-style Transformer 从 4 视图重建 triplane

比一步 LRM 更稳定（多视图条件提供更多 3D 线索）。

## TripoSR (Tochilkin et al., 2024) — Stability AI × Tripo AI

**核心改进**: 改良训练数据 + 架构微调

### 数据策展

Objaverse 中大量模型质量差、光照不自然。TripoSR 做了精细数据策展：
- 筛选高质量子集
- 用更接近真实照片的渲染设置（HDR 环境光、更自然的材质）
- 这比架构改进更关键

### 技术细节

- 架构：LRM 变体（DINO + Transformer + Triplane）
- 速度：A100 上 < 0.5 秒
- 开源：MIT 协议，模型权重 + 代码全开源

### 后继：Stable Fast 3D (SF3D)

更快的推理 + mesh 精化后处理，更适合游戏/AR 场景。

## M-LRM (2024)

**核心改进**: 几何感知位置编码 + 多视图交叉注意力

- 标准 triplane 位置编码 (PE) 无法编码 3D 几何关系
- M-LRM 引入几何感知 PE：将 3D 坐标投影到三个平面的射线信息编码
- 多视图 token 间交叉注意力统一 3D 理解

## LRM-Zero (NeurIPS 2024)

**核心发现**: 不需要真实的 3D 数据！仅用程序化生成的几何体渲染训练，LRM 可以在真实图片上泛化。

这暗示了：3D 重建的核心能力来自"理解形状的拓扑/几何组合方式"，而非学习特定的语义类别。

## 前馈 vs 优化总结

| | 前馈式 (LRM/TripoSR) | 优化式 (DreamFusion/MVDream+SDS) |
|---|---|---|
| 速度 | 亚秒级~数十秒 | 数十分钟~数小时 |
| 质量 | 泛化好，细节不如优化 | Per-instance 精调，细节好 |
| 泛化 | 依赖训练数据分布 | 不需要 3D 训练数据 |
| 典型用途 | 快速预览/批量生成 | 精品制作/创意设计 |
