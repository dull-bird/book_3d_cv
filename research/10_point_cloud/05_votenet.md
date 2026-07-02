# VoteNet: Deep Hough Voting for 3D Object Detection in Point Clouds

**论文**: Qi et al., ICCV 2019
**官方代码**: https://github.com/facebookresearch/votenet

## 核心问题

3D 目标检测：从点云中找到所有物体的 3D bounding box。2D 检测有锚框/区域提议，但 3D 点云是不规则的。

VoteNet 借鉴了霍夫投票的思想：让每个点"投票"出它认为物体中心在哪里，再对投票聚类得到物体提议。

## 关键设计：从点到中心投票

### 网络结构

```
PointNet++ backbone → (N, 256+3) 每点特征 + 坐标
  → Voting Module: MLP 回归 (Δx, Δy, Δz) + 特征残差
  → 投票点 = 原始点 + Δ → (N, 3) votes
  → FPS 采样 K 个 cluster 中心 → Ball query 分组
  → Proposal Module: 对每组 votes 用 shared MLP → (K, proposal_feat)
  → Classification + Box Regression heads
```

### Voting Module

每个种子点学一个指向物体中心的偏移：

```
Δxᵢ = MLP(fᵢ)  # 种子点 i 的中心偏移
```

投票点通过 FPS 聚类后，每组投票生成一个 proposal。

### 损失函数

- 分类损失：Focal Loss（处理类别不平衡）
- 框回归：中心 Δ、尺寸 (l,w,h)、朝向角（F-Net 估计 + 分类 bins）
- 3D IoU: 用 Axis-Aligned or Oriented 3D IoU

## 关键 insight

- 投票机制让远处的点也能对物体中心有贡献——不依赖中心附近的点
- 纯几何方法，不需要 2D 图像的支撑
- 对稀疏点云仍有较好效果

## 实验

- ScanNet V2: mAP@0.25 = 58.6%
- SUN RGB-D: mAP@0.25 = 57.7%
- 在无 RGB 输入的纯几何检测中 SOTA

## 后继发展

- **Group-Free 3D** (Liu et al., ICCV 2021): Transformer decoder 直接预测，无需投票聚类
- **3DETR** (Misra et al., ICCV 2021): Transformer 编码器 + 解码器端到端检测
