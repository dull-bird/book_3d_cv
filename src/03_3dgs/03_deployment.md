# C.3 部署实战

> **目标**：用手机拍一段视频，跑 COLMAP 获取位姿，训练 3DGS 模型，最后在浏览器或 SIBR 查看器中旋转场景。

## 选方案

| 你的需求 | 推荐 |
|---------|------|
| 快速上手、文档好、社区活跃 | nerfstudio（集成了 gsplat） |
| 追求最新精度、想魔改 | 官方 gaussian-splatting 仓库 |
| 只有几张照片（不是视频） | DUSt3R + 3DGS（但稀疏视角质量有限） |
| 显存有限 | Scaffold-GS（更小存储） |

## 方案一：nerfstudio + gsplat（推荐）

nerfstudio 把 COLMAP 预处理和 3DGS 训练打包进一个命令。

### 安装

```bash
pip install nerfstudio
# nerfstudio 会自动安装 gsplat
```

### 数据准备

用手机拍目标物体/场景的**视频**（30 秒左右，绕一圈）→ 用 ffmpeg 抽帧：

```bash
ffmpeg -i input_video.mp4 -q:v 1 -vf fps=2 frames/%06d.jpg
```

### 训练

```bash
# Step 1: COLMAP 预处理（提取特征 + 匹配 + SfM + 去畸变）
ns-process-data images --data frames/ --output-dir colmap_output/

# Step 2: 训练 3DGS（默认 gsplat 后端）
ns-train splatfacto --data colmap_output/
```

`splatfacto` 是 nerfstudio 的 3DGS 变体实现，基于 gsplat（nerfstudio 团队自己写的 CUDA 光栅化器）。

### 查看结果

```bash
# nerfstudio 自带 Web 查看器
ns-viewer --load-config outputs/colmap_output/splatfacto/YYYY-MM-DD_HHMMSS/config.yml
```

浏览器打开 `localhost:7007`，可以实时旋转视角、缩放、切换渲染模式。

### 导出

```python
# 导出为 .ply 点云（高斯中心 + RGB）
from nerfstudio.utils.eval_utils import eval_setup
config_path = "outputs/.../config.yml"
pipeline = eval_setup(config_path)[1]
pipeline.model.export_gaussians("output_gaussians.ply")
```

## 方案二：官方 gaussian-splatting 仓库

适合需要完全控制训练过程的场景。

### 安装

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
```

### COLMAP 预处理

```bash
# 官方仓库需要手动跑 COLMAP
# 或使用 convert.py 脚本
python convert.py -s <location> [--resize]
```

### 训练

```bash
python train.py -s <path_to_COLMAP_data> -m <output_path>
```

关键参数：
- `--iterations 30000`：默认训练轮数
- `--sh_degree 3`：球谐阶数（0=漫反射, 3=完整 view-dependent color）
- `--densify_until_iter 15000`：密度控制停止迭代
- `--lambda_dssim 0.2`：D-SSIM 损失权重

### 查看

```bash
# SIBR 查看器（官方提供的实时交互工具）
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m <output_path>
```

## 训练时间参考

| 场景规模 | GPU | 训练时间 | 模型大小 |
|---------|-----|---------|---------|
| 室内小场景（~200 张图） | RTX 3090 | ~20 min | ~300 MB |
| 室外大场景（Mip-NeRF360, ~300 张图） | A100 | ~40 min | ~700 MB |
| 手机视频（100 帧） | RTX 4060 8GB | ~30 min | ~200 MB |

## 从训练好的模型渲染新视角

```python
import torch
from gaussian_renderer import render
from scene import GaussianModel

# 加载训练好的高斯
gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply("output_gaussians.ply")

# 从任意相机位姿渲染
from utils.camera import Camera
viewpoint = Camera(
    R=torch.tensor([...]),  # 3x3 旋转矩阵
    T=torch.tensor([...]),  # 3x1 平移向量
    FoVx=60.0, FoVy=45.0,   # 视场角
    image_height=720, image_width=1280,
)

render_pkg = render(viewpoint, gaussians, pipe_params)
rendered_image = render_pkg["render"]  # (3, H, W) tensor
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| COLMAP 匹配失败 | 场景纹理太少（白墙） | 放一些带纹理的参照物 |
| 训练后渲染有"浮尘" | 密度控制没清理干净的空洞高斯 | 增大 densification 的梯度阈值 |
| 锯齿严重 | 分辨率变化时未做抗锯齿 | 用 Mip-Splatting |
| 显存不足 | 高斯数量太多 | 用 Scaffold-GS 或降低分辨率 |
| 新视角有空洞 | 训练视角覆盖不足 | 补拍该角度的照片重新训练 |
