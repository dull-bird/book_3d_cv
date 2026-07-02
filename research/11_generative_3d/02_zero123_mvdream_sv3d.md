# 多视图先验：Zero-1-to-3、MVDream、SV3D

## Zero-1-to-3 (Liu et al., ICCV 2023)

**核心思想**: 把 Stable Diffusion 微调为"视角条件图像生成器"。输入单张图片 + 目标相机位姿 (R, T)，输出该视角的图片。

**方法**:
- 在 Objaverse 上微调 SD：输入 (input_img, R, T) → 输出新视角图片
- 条件注入：输入图片通过 CLIP 编码，相机位姿通过 MLP 投影后拼接到时间步嵌入
- 训练后，可以当作"新视角 oracle"：给定任意 (image, pose) → 生成对应视角

**3D 重建**: 有了 Zero-1-to-3，用 SDS/SJC 做 3D 重建：每步采样一个视角，Zero-1-to-3 生成目标视角的监督信号。

**局限**: 各视角生成的图片不一定一致（视角间独立采样），导致几何不一致。

## MVDream (Shi et al., 2023)

**核心思想**: 一个扩散模型同时生成 4 张多视图一致的图片。在 Transformer 架构中，不同视角的图像 token 可以互相 attend，学习跨视角一致性。

**关键设计**:
- 3D self-attention: 4 个视角的 latent 在 attention 中互联
- 训练: 在 Objaverse 上用固定 4 个正交相机渲染
- 输出: 给定文本，输出 4 张一致的多视图图片

**效果**: 显著减少 Janus 问题（不同视角出现不同内容），因为 4 个视角共享同一个"3D 理解"。

**3D 重建**: MVDream → 4 视图 → 作为 SDS 的 3D 先验，约束 NeRF/3DGS 优化。

**局限**: 只支持固定文本输入（无图像条件），需要额外的图像→文本步骤。

## SV3D: Stable Video 3D (Voleti et al., ECCV 2024)

**核心思想**: 用视频扩散模型替代图像扩散模型做新视角合成。视频模型天然具有时序一致性 = 视角间一致性。

**方法**:
- 基于 **Stable Video Diffusion (SVD)**（图像→视频的 latent 扩散模型）
- 输入单张图片 + 相机 elevation 角度 → 生成 21 帧环绕新视角视频
- 视频帧天然一致（时间连续性 = 视角连续性）

**特点**:
- 支持指定 elevation（仰角），控制生成视角范围
- 同时输出新视角和 3D mesh（通过 NeuS 重建）
- 两变体：SV3D_u（无 elevation 条件）和 SV3D_p（有 elevation 条件）

**效果**: 新视角合成和 3D 重建双 SOTA（2024 年初），多视图一致性远超 Zeor-1-to-3。

**与 MVDream 对比**:
| | Zero-1-to-3 | MVDream | SV3D |
|---|---|---|---|
| 条件 | 图像+位姿 | 文本 | 图像+elevation |
| 输出 | 单张新视角 | 4 视图 | 21 帧视频 |
| 3D 一致性 | 弱 | 强 | 最强 |
| 重建方法 | SDS | SDS | NeuS |
