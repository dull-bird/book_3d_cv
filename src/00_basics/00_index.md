# 第 0 章 基础篇导读

> 预计总阅读时间：3-4 小时
> 前置知识：无（本篇是全书起点）
> 读完本篇后，你可以：理解相机如何将三维世界变成二维图像、掌握多视图几何的骨架概念，为后续每一个实战模块打下地基。

## 为什么基础篇是必读的

拍摄一张照片时，真实世界是三维的，照片是二维的。三维怎么变成二维的？这个"压扁"的过程就是**投影**。

本书所有的模块——单目深度估计、双目立体匹配、3D Gaussian Splatting——都在回答同一个问题：**如何从二维图像恢复三维信息**。而回答这个问题的前提，是理解相反的路径：三维是怎么"丢"进二维的。

基础篇要搭建的，就是这套从三维到二维、再从二维回到三维的**概念脚手架**。它不涉及任何深度学习模型，只用几何和代数，把相机成像的本质讲清楚。

## 学习目标

读完本篇导读和七个基础章节后，你将能够：

1. 写出一个三维点到像素坐标的完整投影公式 $P = K[R|t]$，并解释每个符号的含义
2. 理解齐次坐标为什么是计算机视觉的"通用语言"
3. 用两幅图像的匹配点计算基础矩阵 F，理解对极约束如何将匹配搜索从二维降到一维
4. 分辨透视相机和仿射相机的区别，知道什么时候该用哪个
5. 掌握 DLT、RANSAC、Bundle Adjustment 等方法的基本原理，理解为什么"优化"贯穿整个 3D 视觉

## 各节概览与估计阅读时间

| 节          | 内容                                | 核心收获                | 阅读时间  |
| ---------- | --------------------------------- | ------------------- | ----- |
| 第 0 章 基础篇导读 | 基础篇的学习路径、目标和后续模块连接             | 知道为什么要先学几何脚手架      | 10 分钟 |
| 第 1 章 相机模型    | 针孔相机、内参 K、畸变模型、透视/仿射/移轴相机      | 理解相机如何把三维点映射到像素 | 40 分钟 |
| 第 2 章 坐标系转换   | 外参 $R,t$、旋转表示、PnP、标定输出物 | 理解相机在世界里的位置和姿态 | 50 分钟 |
| 第 3 章 投影几何    | 齐次坐标、单应矩阵 H、DLT、张氏标定法             | 理解平面棋盘格为什么能标定相机       | 50 分钟 |
| 第 4 章 多视图几何入门 | 对极几何、$F/E$、位姿恢复、三角测量、立体矫正         | 理解两张图如何恢复相对三维       | 55 分钟 |
| 第 5 章 深度表示    | 深度图、视差图、点云、mesh、TSDF、3DGS                        | 知道三维信息有哪些存法和边界      | 40 分钟 |
| 第 6 章 优化基础    | 残差、最小二乘、RANSAC、鲁棒损失、Bundle Adjustment | 理解有噪声时如何求最好解释  | 45 分钟 |
| 第 7 章 3D 相机与传感器 | 双目、结构光、ToF、LiDAR、线激光、选型指标 | 理解深度数据如何由硬件产生      | 45 分钟 |

> 建议按顺序阅读。第 0 章先给出全局地图；第 1 和第 2 章是后续所有内容的基础——第 1 章建立投影直觉，第 2 章建立坐标系直觉；如果你已有一定基础，第 5 章可以快速扫读。第 7 章是后面实战模块的"硬件背景"，建议在进入模块 A/B/C 前通读一遍。

## 本篇如何连接到后续模块

```
基础篇（本篇）
  ├── 相机模型、坐标系转换 ────→ 模块 A「单目深度估计」：一张图推测深度
  │                           需要理解"世界→像素"的投影过程
  │
  ├── 相机模型、坐标变换 ─────→ 模块 C「3D Gaussian Splatting」：从多图重建三维场景
  │                           需要理解相机内外参和投影几何
  │
  ├── 对极几何、多视图 ───────→ 模块 B「双目立体匹配」：两张图恢复深度
  │                           需要理解对极约束和视差-深度关系
  │
  ├── DLT、RANSAC、BA ────────→ 所有模块：优化和鲁棒估计是贯穿全书的
  │                             底层工具
  │
  └── 3D 相机与传感器 ─────────→ 所有模块：理解数据从哪来、传感器如何影响算法设计
                                 选型直接影响重建质量的上限
```

后续每个模块都会直接调用基础篇的概念。比如：

- 模块 A 讨论"如何从像素坐标反推深度"时，用的是 $P = K[R|t]$ 的逆过程
- 模块 B 讨论"左右图像如何匹配"时，依赖对极几何把搜索空间从整个图像压缩到一条线
- 模块 C 的相机参数优化中，Bundle Adjustment 直接来自基础篇的优化基础
- **第 7 章（3D 相机与传感器）** 回答了一个实践问题：这些数学工具需要什么样的输入数据？不同的 3D 传感器如何用不同的物理原理实现同一个目标——获取深度？读完硬件，再进入算法模块，你会更清楚"为什么这个算法这样做"。

> [!NOTE]
> **阅读路径建议**：01 → **02** → 03 → 04 → 05 → 06 → **07** → 按兴趣选模块 A/B/C。如果把 01-06 比作"学语法"，07 就是"认识笔"——懂语法才能写文章，但用什么笔决定了你能写多细、写多快。

**基础篇不求贪多，但求扎实。** 每个概念都将反复出现在后面的模块中。如果第一次读不太懂某个推导，先记住公式的"人话版本"和它在实践中怎么用，后面自然会越来越清晰。

## 预备知识

本篇是全书起点，只假设你具备：

- 高中数学（向量、矩阵乘法的基本概念）
- 会写简单的 Python 代码
- 对"相机拍照"有常识性理解

**不需要**任何相机标定、射影几何或 3D 重建的预备知识。每一个概念都会从零开始解释。

---

## Mini Case：用手机拍两张照片，手动算桌子的距离

> 这个案例贯穿基础篇第 1-7 章的知识。完成基础篇学习后，你应该能独立完成这个任务。

### 你需要

- 一部手机（拍照）
- Python + OpenCV + NumPy

### 步骤概览

1. **拍照**：用手机从两个不同位置拍同一张桌子，两张照片
2. **标定**：用棋盘格标定手机相机，得到内参 K（对应 §2 坐标系转换）
3. **找对应点**：在两张照片中手动选取至少 8 对同名点（桌子角、杯子边缘等）
4. **算基础矩阵**：用 8 点法 + RANSAC 从对应点算 F（对应 §4 多视图几何）
5. **恢复位姿**：从 F 和 K 得到 E，SVD 分解得到 R 和 t（对应 §4 多视图几何）
6. **三角化**：对每对同名点，用 R、t 恢复 3D 坐标（对应 §4 多视图几何）
7. **算距离**：从 3D 点云的尺度还原桌子的实际宽度

### 代码骨架

```python
import cv2
import numpy as np
from numpy.linalg import svd, inv, norm


# ---- 1. Load images and calibration ----
img1 = cv2.imread("table_01.jpg")
img2 = cv2.imread("table_02.jpg")

# Camera intrinsics from chessboard calibration (H&Z section 6.1)
K = np.array([[1200.0, 0.0, 960.0],
              [0.0, 1200.0, 540.0],
              [0.0, 0.0, 1.0]])  # Replace with your calibrated K

# ---- 2. Detect and match features ----
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match features and apply ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# ---- 3. Find Fundamental matrix with RANSAC (H&Z section 9.2, p.245) ----
F, mask = cv2.findFundamentalMat(
    pts1, pts2, method=cv2.FM_RANSAC,
    ransacReprojThreshold=1.0, confidence=0.99)
pts1_in = pts1[mask.ravel() == 1]
pts2_in = pts2[mask.ravel() == 1]
print(f"{len(pts1_in)} inlier matches found")

# ---- 4. Compute Essential matrix and recover pose (H&Z section 9.6, p.257-258) ----
E = K.T @ F @ K
U, S, Vt = svd(E)
# Enforce E constraint: two equal singular values, one zero
E_corrected = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

# Decompose E to get R and t (4 possible solutions)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
R1 = U @ W @ Vt
R2 = U @ W.T @ Vt

if np.linalg.det(R1) < 0:
    R1 *= -1
if np.linalg.det(R2) < 0:
    R2 *= -1

t = U[:, 2]  # Translation (up to scale)

# Build camera matrices
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = K @ np.hstack([R1, t.reshape(3, 1)])

# ---- 5. Triangulate 3D points (H&Z section 10.1, p.262-264) ----
points_3d_h = cv2.triangulatePoints(P1, P2,
                                     pts1_in[:20].T,
                                     pts2_in[:20].T)
points_3d = points_3d_h[:3] / points_3d_h[3]  # Homogeneous -> Euclidean

# ---- 6. Estimate real-world scale ----
# Compute distances between 3D points
distances = []
for i in range(len(points_3d.T)):
    for j in range(i + 1, len(points_3d.T)):
        dist = norm(points_3d[:, i] - points_3d[:, j])
        distances.append(dist)

print(f"3D point cloud has {len(points_3d.T)} points")
print(f"Pairwise distances (unscaled): {np.sort(distances)[::-1][:5]}")

# ---- 7. Recover scale from a known measurement ----
# Measure one known distance (e.g., table width = 1.2 meters)
# known_pair = (point_i, point_j)  # Find which points correspond
# scale = 1.2 / norm(points_3d[:, known_pair[0]] - points_3d[:, known_pair[1]])
# points_3d_scaled = points_3d * scale
# print(f"Recovered scale factor: {scale:.3f}")
# print(f"Table width from 3D: {norm(points_3d_scaled[:, known_pair[0]] - points_3d_scaled[:, known_pair[1]]):.3f} m")
```

> 这个案例的核心启发是：从两张手机照片恢复桌子的 3D 尺寸，**不需要激光雷达、不需要深度传感器、不需要昂贵的设备**——只需要基础的射影几何知识和 Python。你的手机相机内参 K 是标定一次就复用的；SIFT 匹配是自动的；RANSAC 帮你剔除错配；几何优化如果不用也可以得到不错的结果（尤其是用了 8 点法 + 归一化之后）。实际操作中，最大的误差来源通常是**对应点的精度**和**尺度恢复**——这也是后续实战模块要解决的问题。
