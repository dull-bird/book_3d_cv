# 生成式 3D：文献地图

## 两大范式

```
优化式（Optimization-based）                     前馈式（Feed-forward）
  DreamFusion (2022)                               LRM (2024)
    ↓                                                 ↓
  Magic3D (2023)                                   Instant3D (2024)
    ↓                                                 ↓
  ProlificDreamer (2023)                           TripoSR (2024)
    ↓                                                 ↓
  DreamGaussian (2023)                             M-LRM (2024)
    ↓                                                 ↓
  [需数十分钟到数小时/场景]                         [亚秒级到数十秒/场景]
```

## 里程碑论文

### 优化式：SDS 范式

1. **DreamFusion** (Poole et al., ICLR 2023): 提出 SDS（Score Distillation Sampling），无需 3D 训练数据，用 2D 扩散模型监督 NeRF 优化
2. **Magic3D** (Lin et al., CVPR 2023): 两阶段（NeRF 粗 → Mesh 精），高分辨率纹理
3. **ProlificDreamer** (Wang et al., NeurIPS 2023): VSD（Variational Score Distillation）替代 SDS，解决过度平滑和色彩过饱和
4. **DreamGaussian** (Tang et al., ICLR 2024): 用 3DGS 替代 NeRF，优化几分钟到高质量 mesh

### 多视图先验

5. **Zero-1-to-3** (Liu et al., ICCV 2023): 视角条件扩散模型，单图 → 新视角合成
6. **MVDream** (Shi et al., 2023): 多视图一致扩散模型，解决 Janus 问题
7. **SV3D** (Voleti et al., ECCV 2024): 基于 Stable Video Diffusion 的新视角合成 + 3D 生成

### 前馈式：LRM 家族

8. **LRM** (Hong et al., ICLR 2024): 首个大规模重建模型，Transformer 直接预测 triplane NeRF
9. **Instant3D** (Li et al., ICLR 2024): 稀疏视角生成 + LRM 重建，~20 秒
10. **TripoSR** (Tochilkin et al., 2024): LRM + 改良训练数据，亚秒级，MIT 开源
11. **M-LRM** (2024): 多视图 LRM，几何感知位置编码
12. **LRM-Zero** (NeurIPS 2024): 纯合成数据训练的 LRM

### 文本到 3D 前馈

13. **Instant3D** (Li et al., IJCV 2024): 直接从文本到 triplane，< 1 秒，无需多视图中间步骤
