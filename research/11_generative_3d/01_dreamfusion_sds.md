# DreamFusion & Score Distillation Sampling (SDS)

**论文**: Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion", ICLR 2023
**项目页面**: https://dreamfusion3d.github.io/

## 核心问题

训练 3D 生成模型需要海量 3D 数据（标注的 mesh/点云），而 3D 数据极度稀缺。能否只用 2D 扩散模型生成 3D？

## 核心洞察：用 2D 扩散模型监督 3D 优化

不需要训练 3D 扩散模型。用一个可微的 3D 表示（NeRF/3DGS），从文本描述渲染出图，用预训练 2D 扩散模型判断"渲染图是否像文字描述"。

## Score Distillation Sampling (SDS)

### 扩散模型回顾

给定图像 x 和噪声 ϵ，去噪网络 ϵ̂_ϕ(x_t; y, t) 预测添加的噪声。给定干净图像的梯度方向：

```
∇_x log p(x | y)
```

### SDS 公式

对可微 3D 表示的参数 θ（如 NeRF 的 MLP 权重），渲染图 x = g(θ)，SDS 的梯度：

```
∇_θ L_SDS = E_{t,ϵ} [w(t) (ϵ̂_ϕ(x_t; y, t) - ϵ) ∂x/∂θ]
```

直觉：ϵ̂_ϕ(x_t; y, t) - ϵ 是从噪声到干净图像的"方向向量"。通过链式法则回传，告诉 3D 参数 θ 如何更新使得渲染结果更像文字 y 描述的样子。

### 关键实现细节

- **随机相机采样**：每步随机采样相机位置（球面上均匀分布），鼓励多视图一致性
- **Classifier-free guidance**: ω = 100（高于图像的 7.5），增加引导强度
- **渲染分辨率**: 64×64 训练，逐步增加
- **Imagen 作为骨干**: 但后续工作多用 Stable Diffusion

## 局限性（催生后续工作）

1. **Janus 问题**：不同视角生成不同内容（如正面看是个人，背面也有张脸）
2. **过度平滑**：SDS 来自 mode-seeking 行为，生成结果缺乏细节
3. **慢**：NeRF 渲染 + 扩散模型去噪，每个迭代涉及梯度计算，数十分钟到数小时
4. **颜色过饱和**：高 CFG 权重导致

## SDS 的数学本质

SDS 实际上最小化的是一个 KL 散度的变分上界，相当于让 3D 渲染的分布在扩散模型学到的图像分布中 mode-seeking。
