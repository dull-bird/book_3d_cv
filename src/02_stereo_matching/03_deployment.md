# B.3 部署实战

> **目标**：用 CREStereo 跑一对手机拍的双目照片，理解输出格式差异，知道什么时候选什么模型。

## 选模型

| 你的需求 | 推荐 |
|---------|------|
| 两张标定好的矫正图，追求精度 | CREStereo |
| 两张任意图，未标定，快出结果 | DUSt3R |
| 视频流，需要实时 | RAFT-Stereo Slow-Fast（~26 FPS） |
| 几十张图的大场景 3D 重建 | MVSNet 或 COLMAP + 深度融合 |

## CREStereo

### 安装

```bash
git clone https://github.com/megvii-research/CREStereo
cd CREStereo
pip install -r requirements.txt
```

### 下载模型

```bash
mkdir -p weights
# 从项目 Release 页面下载预训练权重到 weights/
# crestereo_eth3d.pth 或 crestereo_sceneflow.pth
```

### 推理

```python
import cv2
import torch
import numpy as np
from crestereo import CREStereo

model = CREStereo(
    max_disp=256,  # 最大视差范围
    mixed_precision=False,
)
model.load_state_dict(torch.load('weights/crestereo_eth3d.pth'))
model.cuda().eval()

# 读左右图（必须是矫正过的！）
imgL = cv2.imread('left.jpg')
imgR = cv2.imread('right.jpg')

# 转为 tensor, BGR→RGB, normalize
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

imgL_t = preprocess(imgL)
imgR_t = preprocess(imgR)

with torch.no_grad():
    disp = model(imgL_t, imgR_t)  # (1, H, W), 视差图（像素）

# 视差 → 深度: Z = f*b / d
# 假设你的双目对已标定
fx = 525.0   # 像素焦距（从你的 K 矩阵中取）
baseline = 0.12  # 基线（米）
depth = (fx * baseline) / (disp.squeeze().cpu().numpy() + 1e-6)

# 可视化
depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
cv2.imwrite('depth.png', depth_vis.astype(np.uint8))
```

> [!CAUTION]
> CREStereo 假设输入是**矫正过的双目对**——左右图的对应点在同一行上。如果你用手机随意拍的两张照片（没有矫正），结果会非常差。先用 `cv2.stereoRectify` 做矫正。

## DUSt3R（不需要标定的方案）

DUSt3R 的独特优势：**不需要标定、不需要矫正、不需要相机参数。** 两张任意角度拍的照片直接输入。

```python
import torch
from dust3r.model import AsymmetricCroCo3DStereo

model = AsymmetricCroCo3DStereo.from_pretrained(
    'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
).cuda().eval()

# 读两张任意图
from dust3r.utils.image import load_images
imgs = load_images(['view1.jpg', 'view2.jpg'], size=512)

with torch.no_grad():
    output = model(imgs[0], imgs[1])
    # output['pts3d']: (H, W, 3) 两组点图
    # output['conf']: 置信度（哪些像素匹配得好）

pts3d_1 = output['pts3d'][0]  # 第一张图的点图
conf = output['conf'][0]      # 置信度

# 用置信度过滤低质量点
valid = conf > 0.5
print(f"Valid points: {valid.sum()} / {valid.numel()}")
```

> DUSt3R 输出的是 3D 点图（不是视差图也不是深度图），每个像素对应一个 $(X,Y,Z)$ 坐标。多视角时需要用全局对齐把 pairwise 点图拼到一个坐标系中。

## 精度 vs 速度

| 模型 | 推理时间（GPU） | 精度（Middlebury） | 需要标定 |
|------|---------------|-------------------|---------|
| RAFT-Stereo (standard) | ~130 ms | 4.74% bad-2px | 是 |
| RAFT-Stereo (Slow-Fast) | ~50 ms | 接近 standard | 是 |
| CREStereo | ~200 ms | 3.71% bad-2px | 是 |
| DUSt3R | ~1 s | 无 Middlebury 可比指标 | **否** |

> **选择建议**：有标定好的双目对时用 CREStereo（精度最高）。需要实时处理用 RAFT-Stereo Slow-Fast。没有任何标定信息时用 DUSt3R（最方便，但精度和速度都不如专用双目模型）。
