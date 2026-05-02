# A.3 部署实战

> **目标**：能在自己电脑上跑 Depth Anything V2 和 MoGe，理解它们的输出格式差异，知道什么时候选哪个。

## 选模型的三条经验法则

| 你的需求 | 推荐 | 原因 |
|---------|------|------|
| 只要"哪近哪远"（景深模拟、背景虚化） | Depth Anything V2 Small | 25M 参数，CPU 可跑，relative depth 足够 |
| 需要 metric 深度（机器人、3D 重建） | Depth Anything V2 Metric 版 或 MoGe | 输出带物理单位的深度 |
| 艺术风格图、极端非自然场景 | Marigold | 扩散模型先验空间更广 |
| 精度第一，速度无所谓 | Depth Anything V2 Giant | 1.3B 参数，最强 zero-shot 精度 |

## Depth Anything V2

### 安装

```bash
pip install torch torchvision
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

### 下载模型

从 HuggingFace 下载预训练权重。四个版本：

```python
# metric_depth_estimation.py 中会自动下载，也可以手动指定
# Small:  https://huggingface.co/depth-anything/Depth-Anything-V2-Small
# Base:   https://huggingface.co/depth-anything/Depth-Anything-V2-Base
# Large:  https://huggingface.co/depth-anything/Depth-Anything-V2-Large
# Giant:  https://huggingface.co/depth-anything/Depth-Anything-V2-Giant
```

### 推理：relative depth

```python
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# 选模型
model = DepthAnythingV2(
    encoder='vits',      # vits / vitb / vitl / vitg
    features=64,         # 64 / 128 / 256 / 384
    out_channels=[48, 96, 192, 384],
)

# 加载权重
model.load_state_dict(torch.load(
    'checkpoints/depth_anything_v2_vits.pth', 
    map_location='cpu'
))
model.eval()

# 读图、推理
img = cv2.imread('your_photo.jpg')
h, w = img.shape[:2]
# 模型期望 518x518 输入
img_resized = cv2.resize(img, (518, 518))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()

with torch.no_grad():
    depth = model(img_tensor)  # (1, 518, 518), affine-invariant

# 恢复原图尺寸
depth = torch.nn.functional.interpolate(
    depth.unsqueeze(1), size=(h, w), mode='bilinear'
).squeeze().numpy()

# 可视化（归一化到 0-255）
depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
cv2.imwrite('depth_vis.png', depth_vis.astype(np.uint8))
```

### 推理：metric depth

```python
from depth_anything_v2.dpt import DepthAnythingV2

# Metric 版本需要不同配置
model = DepthAnythingV2(
    encoder='vitl',
    features=256,
    out_channels=[256, 512, 1024, 1024],
    max_depth=20.0,  # 最大深度（米），根据场景调整
)

model.load_state_dict(torch.load(
    'checkpoints/depth_anything_v2_metric_vkitti_vitl.pth',
    map_location='cpu'
))
model.eval()

# 推理同上，但输出是 metric 深度（米）
with torch.no_grad():
    depth = model(img_tensor)  
# depth 的单位是米，值范围取决于 max_depth 参数
print(f"Depth range: {depth.min():.2f}m to {depth.max():.2f}m")
```

### 理解输出

| 版本 | 输出 | 数值含义 | 单位 |
|------|------|---------|------|
| Relative（默认） | $(H, W)$ array | 相对远近（大 = 远），只保持顺序关系 | 无量纲 |
| Metric（微调版） | $(H, W)$ array | 沿光轴的物理距离 | 米 |

> [!CAUTION]
> **Metric 版本不是零成本的。** 它是在特定域（如 KITTI/室外或 NYU/室内）上微调的。如果你用它拍室内照片但用的是 outdoor metric 版，尺度会偏。对泛化要求高时，先跑 relative 版本，再手动对齐尺度。

## MoGe

MoGe 的优势是直接输出 **3D 点图 + 深度图 + 相机内参**，不需要额外 step。

### 安装

```bash
pip install moge
# 或从源码
git clone https://github.com/microsoft/MoGe
cd MoGe
pip install -e .
```

### 推理

```python
import cv2
import torch
import numpy as np
from moge.model import MoGeModel

model = MoGeModel.from_pretrained('Ruicheng/moge-rtnlk').eval()

img = cv2.imread('your_photo.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 推理（自动处理 resize）
with torch.no_grad():
    output = model.infer(img_rgb)
    # output['points']   : (H, W, 3)  affine-invariant 3D point map
    # output['depth']    : (H, W)     metric depth（米）
    # output['intrinsics']: (3, 3)    recovered camera intrinsics
    # output['fov']      : float      视场角（度）

print(f"FOV: {output['fov']:.1f}°")
print(f"Depth range: {output['depth'].min():.2f} - {output['depth'].max():.2f} m")
print(f"Recovered intrinsics:\n{output['intrinsics']}")
```

### 理解输出

MoGe 的输出比传统深度模型多了两个维度：

- **3D 点图 `points`**：每个像素对应一个 $(X, Y, Z)$ 坐标（相机坐标系，scale 是 metric 的）
- **相机内参 `intrinsics`**：用约 3ms 从点图中恢复的焦距和主点
- **FOV**：直接给出的视场角

> [!TIP]
> 如果你需要把 MoGe 的深度图转回点云：直接取 `output['points']` 就是对齐好的 3D 坐标。不需要像传统深度图那样手动做 $X = (u-c_x)Z/f_x$ 的反投影。这是 MoGe 点图表示的核心工程优势。

## 精度 vs 速度

在 2024 年的模型上，大致趋势（具体数字因硬件和输入分辨率而异）：

| 模型 | 推理时间（GPU） | Zero-shot 泛化 | 输出类型 |
|------|---------------|---------------|---------|
| Depth Anything V2 Small | ~10 ms | 好 | Relative |
| Depth Anything V2 Large | ~50 ms | 很好 | Relative / Metric |
| Depth Anything V2 Giant | ~200 ms | 最好 | Relative |
| MoGe | ~100 ms | 很好 | **Metric + 点图 + 内参** |
| Marigold (4-step DDIM) | ~1-2 s | 很好（尤其艺术图） | Relative |

> **选择建议**：实际项目中 90% 的情况用 Depth Anything V2 Large metric 版就够。需要相机内参或 3D 点坐标时用 MoGe。处理艺术风格或极度非自然图像时考虑 Marigold。移动端或实时场景用 Depth Anything V2 Small。
