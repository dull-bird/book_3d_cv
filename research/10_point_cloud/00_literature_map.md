# 点云处理：文献地图

## 奠基性工作

- **ICP** (Besl & McKay, 1992): 迭代最近点，点云配准的奠基算法
- **PointNet** (Qi et al., CVPR 2017): 首个直接在原始点云上做深度学习的网络，对称函数(max pooling)保证置换不变性
- **PointNet++** (Qi et al., NeurIPS 2017): 引入层级结构，set abstraction + 多尺度分组，PointNet 的自然进化
- **DGCNN** (Wang et al., ACM TOG 2019): 动态图卷积，EdgeConv 在特征空间建图

## 点云卷积演进

```
PointNet (2017) → PointNet++ (2017) → DGCNN (2019) → KPConv (2019) → Point Transformer (2021) → Stratified Transformer (2022)
```

- **KPConv** (Thomas et al., ICCV 2019): 核点卷积，在 3D 空间定义可变形卷积核
- **Point Transformer** (Zhao et al., ICCV 2021): 自注意力机制用于点云
- **Stratified Transformer** (Lai et al., CVPR 2022): 分层窗口注意力，3D Swin

## 点云配准

- **ICP** (Besl & McKay, 1992): 迭代最近点，局部收敛
- **FPFH + RANSAC** (Rusu et al., 2009): 手工特征 + RANSAC 全局配准
- **FGR** (Zhou et al., ECCV 2016): 快速全局配准，基于 FPfh + 交替优化
- **TEASER++** (Yang et al., IEEE T-RO 2021): 可认证鲁棒配准，容忍 90%+ 外点
- **PointNetLK** (Aoki et al., CVPR 2019): 深度学习的刚体配准
- **GeoTransformer** (Qin et al., ECCV 2022): Transformer 配准 SOTA

## 3D 目标检测

- **VoteNet** (Qi et al., ICCV 2019): 霍夫投票 + PointNet++，纯点云检测
- **PointPillars** (Lang et al., CVPR 2019): 将点云转为柱体伪图像，速度极快
- **CenterPoint** (Yin et al., CVPR 2021): 基于中心的检测器

## 自监督学习（2023-2024 新方向）

- **Point-MAE** (Pang et al., ECCV 2022): 掩码自编码器
- **Point-BERT** (Yu et al., CVPR 2022): BERT 风格预训练
- **I2P-MAE** (Zhang et al., CVPR 2023): 图像引导的点云预训练
