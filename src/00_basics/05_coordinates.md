# 05 坐标系转换：从世界到像素的完整旅程

> 预计阅读时间：35 分钟
> 前置知识：本篇第 01 节（相机模型）、第 02 节（投影几何）、第 04 节（深度表示）
> 读完本节后，你可以：写出世界坐标系→相机坐标系→图像坐标系→像素坐标系的完整变换链，理解齐次坐标如何让每一步都变成线性运算，在 OpenCV、OpenGL、Unity 三种坐标约定之间做转换。

---

## 5.1 第一阶：直观理解

### 5.1.1 一个场景

你在玩 VR 游戏，伸手去抓一个虚拟杯子。三个不同坐标系在你毫不知情的情况下高速运转：

- 杯子的模型是美术同学做的——它存储为**世界坐标系**下的坐标，比如 $(3, 0.8, 2)$，表示它在虚拟房间的角落往右 3 米、高 0.8 米、往前 2 米。
- 你头盔上的摄像头看到的杯子位置，是**相机坐标系**下的——以摄像头光心为原点。从摄像头的视角看，杯子在 $(0.5, -0.3, 2.1)$——右前方偏下约 2 米处。
- 最终显示在屏幕上的位置是**像素坐标系**下的——1920×1080 的显示器上，杯子应该画在第几行第几列的 pixel？

没有坐标系转换，这三者完全无法对话。坐标系转换就是给每个参考系之间架桥——四座桥，四个变换，把同一个杯子从"房间里的模型"一路护送到底，直到屏幕上那个发光的像素。

### 5.1.2 核心直觉：杯子的四次"翻译"

还是这个 VR 抓杯子的场景。跟踪杯子上同一个点——比如说杯口的中心——看看它在每一步变换中经历了什么：

1. **世界坐标系 → 相机坐标系**：杯子在房间里的位置 $(3, 0.8, 2)$，用你头盔摄像头的视角重新表达，变成了 $(0.5, -0.3, 2.1)$。这一步只需要旋转和平移——**刚体变换**，6 个自由度。它回答的问题是"杯子在摄像头的正前方多远？"——跟房间的墙角无关。

2. **相机坐标系 → 图像坐标系**：摄像头把三维世界"压扁"到成像平面上。杯口中心的 Z 坐标（2.1 米，即深度）消失，只剩下 $(x, y)$——一个连续的二维位置。这是**透视投影**——不可逆的一步。你丢了深度，从此只知道杯子"在视野的哪个方向"，不知道它"有多远"。

3. **图像坐标系 → 像素坐标系**：CMOS 传感器把连续的光信号离散成像素格。$(x, y)$ 乘以每毫米多少像素，再加上图像中心点的像素坐标（主点）——变成 $(u, v)$，比如 $(960, 540)$——第 540 行第 960 列。这是**采样和缩放**。

4. 最后，GPU 拿着这个 $(u, v)$ 在屏幕上画出杯子。

整个链条合在一起，就是 $P = K[R|t]$——整本书最重要的公式之一。本章要做的，就是把每一步拆开来看清楚。

### 5.1.3 技术全景

```mermaid
mindmap
  root((Coordinate Systems))
    World Frame
      Arbitrary origin
      3 DOF translation
      Z-up or Y-up convention
    Camera Frame
      Origin at camera center C
      Z = optical axis forward
      X right Y down (OpenCV)
    Image Frame
      2D continuous coordinates
      Origin at principal point
      Unit: mm or normalized
    Pixel Frame
      2D discrete coordinates
      Origin at top-left (0,0)
      Unit: pixel integer
    Transforms
      World -> Camera R t 6 DOF
      Camera -> Image perspective K
      Image -> Pixel affine scaling
```

### 5.1.4 Mini Case：跟踪一个点的完整旅程

```python
import numpy as np

# --- Define the scene ---
# A 3D point in world coordinates: a cup on the table
X_world = np.array([3.0, 0.8, 1.5, 1.0])  # homogeneous (H&Z section 2.2.1, p.26)

# Camera pose (where the camera is in the world)
R = np.array([
    [0.0,  0.0, -1.0],
    [-1.0, 0.0,  0.0],
    [0.0,  1.0,  0.0]
])  # rotation matrix
C = np.array([1.0, 0.5, 0.0])  # camera center in world coordinates

# Camera intrinsics
K = np.array([
    [525.0,   0.0, 320.0],
    [  0.0, 525.0, 240.0],
    [  0.0,   0.0,   1.0]
])

# --- Step 1: World -> Camera (rigid body, H&Z 6.6, p.156) ---
t = -R @ C   # t = -R*C (H&Z p.156, Eq 6.6)
Rt = np.column_stack([R, t])  # 3x4 extrinsic matrix
X_cam = Rt @ X_world  # X_cam is 3x1: (X_c, Y_c, Z_c) in camera frame
print(f"Camera coord: ({X_cam[0]:.2f}, {X_cam[1]:.2f}, {X_cam[2]:.2f})")

# --- Step 2: Camera -> Image (perspective projection, H&Z 6.1, p.154) ---
P = K @ Rt  # Full camera matrix (H&Z 6.8, p.156)
x_image = P @ X_world  # homogeneous pixel coord: (u*h, v*h, h)

# --- Step 3: Image -> Pixel (dehomogenize, H&Z 6.1, p.155) ---
u, v = x_image[0] / x_image[2], x_image[1] / x_image[2]
print(f"Pixel coord: ({u:.1f}, {v:.1f})")

# --- Recap: the entire chain in one line ---
x_pixel = K @ Rt @ X_world
print(f"One-liner verification: ({x_pixel[0]/x_pixel[2]:.1f}, "
      f"{x_pixel[1]/x_pixel[2]:.1f})")
```

这段代码追踪了一个三维点从世界坐标系到像素坐标系的完整旅程——四个坐标系，三个变换，一步到位。每个变换都是线性的（在齐次坐标下），整条链就是 $3 \times 4$ 相机矩阵 $P = K[R|t]$ 的一次乘法。

---

## 5.2 第二阶：原理解析

### 5.2.1 变换链全景

```mermaid
flowchart LR
    subgraph W["World Frame O_w"]
        Xw["X_w = (X, Y, Z, 1)^T"]
    end
    subgraph C["Camera Frame O_c"]
        Xc["X_c = R(X_w - C_tilde)"]
    end
    subgraph I["Image Frame"]
        x["x = (fX_c/Z_c, fY_c/Z_c)^T"]
    end
    subgraph Px["Pixel Frame"]
        uv["(u, v) = (alpha_x*x/Z + x0, alpha_y*y/Z + y0)"]
    end
    W -->|"R, t<br/>6 DOF"| C
    C -->|"Perspective<br/>Projection<br/>3D -> 2D lossy"| I
    I -->|"K matrix<br/>Scaling + Offset"| Px
```

每一步的矩阵维度和变换性质：

| 变换 | 矩阵 | 维度 | 自由度 | 可逆？ | H&Z 出处 |
|------|------|------|--------|--------|---------|
| 世界 → 相机 | $[R \mid t]$ | $3 \times 4$ | 6 (3 旋转 + 3 平移) | 是（刚体变换） | 6.6, p.156 |
| 相机 → 图像 | 透视投影 | $3 \times 4$ | 0（几何过程，无自由参数） | **否**——Z 信息丢失 | 6.1, p.154 |
| 图像 → 像素 | $K$ | $3 \times 3$ | 3-5（内参个数） | 是（仿射变换） | 6.4, p.155 |

### 5.2.2 第一步：世界 → 相机（刚体变换）

这是最直观的一步：把一个点从一个坐标系平移到另一个坐标系。相机在世界空间中有一个位置 $\tilde{C}$（相机中心的 world 坐标）和一个朝向 $R$（旋转矩阵）。

两种等价写法（H&Z section 6.1, p.155-156）：

$$X_{\text{cam}} = R(X_{\text{world}} - \tilde{C}) \quad \text{（直观：先平移世界点到相机原点，再旋转）}$$

$$X_{\text{cam}} = R X_{\text{world}} + t \quad \text{其中 } t = -R\tilde{C} \quad \text{（H\&Z 6.6, p.156）}$$

第二种写法在齐次坐标下最方便——直接拼成 $3 \times 4$ 矩阵 $[R \mid t]$：

$$\begin{bmatrix} X_c \cr Y_c \cr Z_c \end{bmatrix} = \begin{bmatrix} R_{11} & R_{12} & R_{13} & t_1 \cr R_{21} & R_{22} & R_{23} & t_2 \cr R_{31} & R_{32} & R_{33} & t_3 \end{bmatrix} \begin{bmatrix} X_w \cr Y_w \cr Z_w \cr 1 \end{bmatrix}$$

6 DOF：3 个旋转角度（R 的正交约束下） + 3 个平移分量。这是整个变换链中最"贵"的一步——相机位姿估计（PnP、标定、SLAM）主要就是在估计这 6 个参数。

### 5.2.3 第二步：相机 → 图像（透视投影——不可逆的一步）

这是**中心投影**——三维点通过小孔投射到二维成像平面。它是整个链中唯一不可逆的步骤，因为 Z 坐标在这个过程里被"压缩"了（H&Z section 6.1, p.154）：

$$(X_c, Y_c, Z_c)^T \longmapsto \left(\frac{f X_c}{Z_c}, \frac{f Y_c}{Z_c}\right)^T$$

其中 $f$ 是焦距（毫米）。除以 $Z_c$ 这一步就是"透视"的来源——同样的物体，远了三倍，在图像上就只剩三分之一大。

在齐次坐标下，这一步可以写成矩阵形式（H&Z 6.2, p.154）：

$$x = \begin{bmatrix} f & 0 & 0 & 0 \cr 0 & f & 0 & 0 \cr 0 & 0 & 1 & 0 \end{bmatrix} X_{\text{cam}} = \text{diag}(f, f, 1) [I \mid 0] \, X_{\text{cam}}$$

其中 $x = (u, v, w)^T$ 是齐次坐标，真实的图像坐标是 $(u/w, v/w)$。

**关键洞察**：从 $3 \times 4$ 的原生投影矩阵 $[I \mid 0]$ 可以看出——第四列全是 0。这意味着相机中心（第四列对应的世界坐标点 $(0,0,0,1)$ 在相机坐标系下的齐次形式）映射到 $(0,0,0)^T$——一个齐次坐标为零的向量，不对应任何有限的图像点。这正好印证了"相机中心本身不能被投影"的物理事实。

### 5.2.4 第三步：图像 → 像素（K 矩阵登场）

图像坐标系里的 $(x, y)$ 是以主点（principal point，光轴与成像平面的交点）为原点、以毫米为单位的。但像素坐标系是以图像左上角为原点、以像素为单位的。两者之间是一个仿射变换（H&Z section 6.1, p.155-157）：

$$\begin{bmatrix} u \cr v \cr 1 \end{bmatrix} = \begin{bmatrix} \alpha_x & 0 & x_0 \cr 0 & \alpha_y & y_0 \cr 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \cr y \cr 1 \end{bmatrix}$$

其中：
- $\alpha_x = f \cdot m_x$：横向焦距（像素单位），$m_x$ 是传感器每毫米的像素数
- $\alpha_y = f \cdot m_y$：纵向焦距（像素单位）
- $x_0, y_0$：主点的像素坐标（通常接近图像中心）
- 最通用的 $K$ 还包含偏斜参数 $s$（H&Z 6.10, p.157），但现代相机 $s \approx 0$

### 5.2.5 完整链：$P = K[R|t]$

将三步合并（H&Z 6.8, p.156）：

$$\boxed{x_{\text{pixel}} = K [R \mid t] X_{\text{world}} = P X_{\text{world}}}$$

展开即：

$$\begin{bmatrix} u \cr v \cr 1 \end{bmatrix} = \begin{bmatrix} \alpha_x & 0 & x_0 \cr 0 & \alpha_y & y_0 \cr 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} R_{11} & R_{12} & R_{13} & t_1 \cr R_{21} & R_{22} & R_{23} & t_2 \cr R_{31} & R_{32} & R_{33} & t_3 \end{bmatrix} \begin{bmatrix} X_w \cr Y_w \cr Z_w \cr 1 \end{bmatrix}$$

$P$ 是一个 $3 \times 4$ 矩阵，秩为 3，有 11 个自由度（5 内参 + 3 旋转 + 3 平移）。这个公式是多视图几何万神殿中的第一神祇——所有下游算法，从三角测量到 BA（Bundle Adjustment），都以它为起点。

**齐次坐标的妙处**：等式 $\lambda x = PX$ 中，尺度因子 $\lambda$ 可以任意缩放。这意味着 $P$ 也只在尺度上可定义——$P$ 和 $kP$（$k \neq 0$）对应同一个投影。因此 $P$ 有 11 个而非 12 个自由度。

### 5.2.6 Code Lens：实现完整变换链

```python
import numpy as np


def world_to_camera(X_world, R, C):
    """World to Camera: rigid body transformation.
    H&Z (6.6, p.156): X_cam = R(X_world - C_tilde)
    Equivalent: X_cam = RX_world + t, where t = -RC_tilde
    """
    t = -R @ C
    return R @ X_world + t


def camera_to_image(X_cam, f=1.0):
    """Camera to Image: perspective projection.
    H&Z (6.1, p.154): (x, y) = (f*X/Z, f*Y/Z)
    Returns normalized image coordinates (f=1).
    """
    x = X_cam[0] / X_cam[2]
    y = X_cam[1] / X_cam[2]
    return x, y


def image_to_pixel(x, y, K):
    """Image to Pixel: affine scaling with K matrix.
    H&Z (6.9, p.157): (u,v) = (alpha_x * x + x0, alpha_y * y + y0)
    """
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v


def full_projection(X_world, K, R, C):
    """Complete chain: World -> Camera -> Image -> Pixel.
    H&Z (6.8, p.156): x_pixel = K[R|t] X_world
    """
    t = -R @ C
    Rt = np.column_stack([R, t])
    P = K @ Rt  # 3x4 camera matrix
    x_homo = P @ X_world
    u = x_homo[0] / x_homo[2]
    v = x_homo[1] / x_homo[2]
    return u, v


# --- Step-by-step trace ---
np.random.seed(42)
X_world = np.array([5.0, 2.0, 10.0])

# Create a random but valid camera pose
from scipy.spatial.transform import Rotation
rot = Rotation.from_euler('xyz', [15, -10, 5], degrees=True)
R = rot.as_matrix()
C = np.array([0.0, 0.0, 0.0])  # camera at world origin

# Standard K
K = np.array([[500.0, 0.0, 320.0],
              [0.0, 500.0, 240.0],
              [0.0, 0.0, 1.0]])

# Step 1
X_cam = world_to_camera(X_world, R, C)
print(f"Camera coord: X_c={X_cam[0]:.2f}, Y_c={X_cam[1]:.2f}, Z_c={X_cam[2]:.2f}")

# Step 2
x, y = camera_to_image(X_cam)
print(f"Image coord (normalized): x={x:.4f}, y={y:.4f}")

# Step 3
u, v = image_to_pixel(x, y, K)
print(f"Pixel coord: u={u:.1f}, v={v:.1f}")

# One-shot verification
u_full, v_full = full_projection(
    np.append(X_world, 1.0), K, R, C
)
print(f"Verification (one-shot): u={u_full:.1f}, v={v_full:.1f}")
```

### 5.2.7 坐标系转换的逆方向：从像素回到世界

理解了正向链后，反向链同样重要——许多任务需要从图像坐标推断三维位置：

| 方向 | 可用信息 | 能恢复什么 | 不能恢复什么 |
|------|---------|-----------|------------|
| Pixel → Image | K 已知 | 连续图像坐标 $(x, y)$ | —（可逆仿射变换） |
| Image → Camera | 无深度信息 | 射线方向：$X/Z = x/f$，$Y/Z = y/f$ | Z（深度）——一条射线上的所有点投影到同一个像素 |
| Camera → World | $R, t$ 已知 | $X_w = R^{-1}(X_c - t) = R^T X_c - R^T t$ | —（可逆刚体变换） |

要点：从像素反推到三维世界时，你只能恢复一条射线（方向知道，深度未知）——而不是一个确定的点。这正是三角测量需要第二台相机的原因。

---

## 5.3 第三阶：部署实战

### 5.3.1 OpenCV 坐标约定

OpenCV 采用右手坐标系（H&Z 全书统一使用）：

| 轴 | 方向 | 备注 |
|----|------|------|
| X | 右 | — |
| Y | 下 | 与像素行号增加方向一致 |
| Z | 前（远离相机） | 光轴方向（"深度"） |

右手定则：右手大拇指指向 X，食指指向 Y，中指自然指向 Z（前方）。这一定则在 OpenCV 的所有函数中一致——从 `solvePnP` 到 `projectPoints` 到 `stereoRectify`。

### 5.3.2 常见坐标系统对比

```mermaid
quadrantChart
    title Coordinate System Comparison
    x-axis "X Direction" --> "Right"
    y-axis "Y Direction" --> "Up vs Down"
    quadrant-1 "OpenCV<br/>X=Right, Y=Down, Z=Forward<br/>Right-Hand"
    quadrant-2 "OpenGL<br/>X=Right, Y=Up, Z=Backward<br/>Right-Hand"
    quadrant-3 "Unity<br/>X=Right, Y=Up, Z=Forward<br/>Left-Hand"
    quadrant-4 "ROS Robot<br/>X=Forward, Y=Left, Z=Up<br/>Right-Hand"
```

**转换速查表**：

| 从 | 到 | 变换 |
|----|----|------|
| OpenCV | OpenGL | $Y \to -Y$，$Z \to -Z$（$X$ 不变） |
| OpenCV | Unity | $Y \to -Y$（$X$ 和 $Z$ 不变——注意左手系） |
| OpenGL | Unity | $Z \to -Z$（绕 X 轴翻转 Z，左手变右手） |
| ROS (robot) | OpenCV | $X \to Z$，$Y \to -X$，$Z \to -Y$（大旋转） |

记住一个检查方法：**在原点放一个正方体，用两种约定分别计算八个顶点的坐标——哪个角在哪个约定下"飞"到了错误的位置，就说明转换错了。**

### 5.3.3 实战：读取相机标定文件，变换 LiDAR 点到相机帧

```python
import numpy as np
import json


def load_calibration(calib_path):
    """Load camera calibration (K) and LiDAR-to-camera extrinsics."""
    with open(calib_path) as f:
        calib = json.load(f)
    K = np.array(calib['K']).reshape(3, 3)
    R_lidar_to_cam = np.array(calib['R_lidar2cam']).reshape(3, 3)
    t_lidar_to_cam = np.array(calib['t_lidar2cam']).reshape(3, 1)
    return K, R_lidar_to_cam, t_lidar_to_cam


def project_lidar_to_image(lidar_points, K, R, t):
    """Project LiDAR points (in LiDAR frame) to image pixel coordinates.

    Steps:
      1. LiDAR frame -> Camera frame: X_cam = R @ X_lidar + t
      2. Camera frame -> Image frame:  x = K @ X_cam (normalized)
      3. Filter points behind camera (Z_cam <= 0)
    H&Z (6.8, p.156): x_pixel = K[R|t] X_lidar
    """
    # Step 1 & 2 combined
    lidar_xyz = lidar_points[:, :3].T  # 3xN
    lidar_homo = np.vstack([lidar_xyz, np.ones(lidar_xyz.shape[1])])  # 4xN
    Rt = np.column_stack([R, t.flatten()])  # 3x4
    P = K @ Rt  # 3x4 camera matrix
    img_homo = P @ lidar_homo  # 3xN

    # Step 3: dehomogenize and filter
    depth = img_homo[2, :]
    valid = depth > 0  # keep only points in front of camera
    u = img_homo[0, valid] / depth[valid]
    v = img_homo[1, valid] / depth[valid]
    points_in_front = lidar_points[valid]

    return u, v, points_in_front, depth[valid]


# --- Example usage ---
# Synthetic LiDAR points (in LiDAR frame)
lidar_pts = np.random.uniform(-10, 10, (1000, 4))

# Load from calibration
K, R_lc, t_lc = load_calibration("calibration.json")

# Project
u_vals, v_vals, pts_3d, depths = project_lidar_to_image(
    lidar_pts, K, R_lc, t_lc
)
print(f"Projected {len(u_vals)} / {len(lidar_pts)} LiDAR points onto image")
print(f"Depth range: [{depths.min():.2f}, {depths.max():.2f}] m")
```

这段代码是多传感器融合的标准操作——自动驾驶中最常见的场景：把 LiDAR 点投影到相机图像上做"着色"（给点云染上图像 RGB），或做"深度补全"（用 LiDAR 稀疏深度引导稠密深度估计）。

---

## 5.4 苏格拉底时刻

1. **如果你把世界坐标系的原点从房间角落移到相机上，所有的三维坐标都变了。但相机拍到的照片不会变——为什么？这暗示了什么关于"绝对坐标"和"相对坐标"的本质差异？**

（提示：照片记录的是三维场景相对于相机的位置——即相机坐标系的 $X_c, Y_c, Z_c$。世界坐标系的原点位置不影响相机与场景之间的相对关系。改变世界原点只是给所有世界坐标加一个偏移量——这个偏移量在 $X_{\text{cam}} = R(X_w - \tilde{C})$ 中被 $\tilde{C}$ 的变化抵消了。这揭示了多视图几何的一个根本事实：**从图像中我们只能恢复相对几何——相机与场景之间、相机与相机之间的相对位置。绝对的世界坐标是一个人类赋予的标签，不是视觉系统能够感知的东西。** 这也是为什么从 E 矩阵恢复平移时，$t$ 的绝对尺度无法确定——你永远不知道相机移动了 1 米还是 10 米，除非有额外的尺度信息（如已知基线长度的双目相机、或已知大小的标定物）。）

2. **齐次坐标让透视投影变成了线性运算。但"除以 Z"这个非线性步骤并没有真的消失——它去了哪里？**

（提示：齐次坐标的把戏是把"除以 Z"推迟到最后一步——投影矩阵 $P$ 的输出是齐次坐标 $(u, v, w)^T$，真实的像素坐标需要做去齐次化 $(u/w, v/w)$。这个 $w$ 恰好等于相机坐标系下的 $Z_c$。所以齐次坐标没有"消灭"非线性——它只是把非线性从计算中间移到了计算末尾，从而让中间的矩阵乘法保持线性。这个优雅的 trick 才使得整条变换链能用一次矩阵乘法完成。）

---

## 5.5 关键论文与文献清单

| 年份 | 文献 | 一句话贡献 |
|------|------|-----------|
| 2004 | Hartley & Zisserman, *Multiple View Geometry*, Ch.6 | $P = K[R|t]$ 的完整推导——世界到像素四坐标系变换链的定义与分解（p.154-157） |
| 2004 | Hartley & Zisserman, *Multiple View Geometry*, Ch.2 | 齐次坐标、变换分层——射影/仿射/相似/欧几里得变换群的不变性层级（p.37-44） |
| 2000 | Z. Zhang, "A Flexible New Technique for Camera Calibration", *IEEE TPAMI* | 张氏标定法——只用棋盘格平面从多张图像同时恢复 $K$ 和每帧的 $[R|t]$ |

---

## 5.6 实操练习

1. **亲手走一遍变换链**：在房间角落里放一个虚拟相机（$C = (0,0,0)$，看向 Z 轴），在 $(3, 0.5, 5)$ 处放一个点。用手算（或用 Python 逐步骤算）这个点从世界坐标系走到像素坐标系的完整过程。验证最终结果和直接用 $P = K[R|t]$ 一次算出来的结果完全相同。
2. **坐标约定陷阱实验**：把一个 OpenGL 渲染的立方体的顶点坐标手动转换到 OpenCV 约定（$Y \to -Y$, $Z \to -Z$），然后用 OpenCV 的 `projectPoints` 投影到图像上。如果不做转换直接投影——立方体出现在图像的什么位置？为什么？
3. **逆推射线**：已知相机内参 $K$ 和图像上的一个像素 $(u,v)$，计算这条像素对应的三维射线在相机坐标系下的方向向量。在空间中画出这条射线——验证它确实穿过相机中心 $(0,0,0)$ 和成像平面上的 $(u - c_x, v - c_y, f)$。

---

## 5.7 延伸阅读

- 本书内：[[01 相机模型]] · [[02 投影几何]] · [[04 深度表示]]
- H&Z 原书 Ch.6.1：相机模型完整推导——从针孔投影到 $P = K[R|t]$（p.154-157）
- H&Z 原书 Ch.6.2.4：RQ 分解——从已知 $P$ 矩阵中恢复 $K$ 和 $R$（p.163-165）
- H&Z 原书 Ch.2.4：变换分层——射影/仿射/相似/欧几里得的不变性，理解"绝对坐标"与"相对坐标"的本质（p.37-44）
- OpenCV Camera Calibration Tutorial: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
