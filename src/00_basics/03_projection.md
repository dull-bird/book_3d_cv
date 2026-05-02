# 第 3 章 投影几何：单应矩阵与张氏标定

> [!NOTE]
> **预计阅读时间**：50 分钟 · **前置知识**：第 1 章（相机模型）、第 2 章（坐标系转换）
>
> 第 2 章说标定会输出 $K$、畸变和每张图的外参，但没有解释“为什么棋盘格照片能做到这件事”。本章回答这个问题：**当世界点都在一个平面上时，投影会简化成单应矩阵 $H$；张氏标定法正是利用多个平面单应矩阵反推出相机内参。**

---

## 3.1 本章目标

第 2 章的完整投影链是：

$$\lambda x = K[R|t]X_w$$

如果 $X_w$ 是任意 3D 点，这就是一个 $3 \times 4$ 投影矩阵。但棋盘格有一个特殊性质：所有角点都在同一个平面上。我们可以把棋盘格坐标系放在棋盘格平面上，让每个角点满足：

$$X_w=(X,Y,0,1)^T$$

代入完整投影：

$$\lambda x = K[r_1\ r_2\ r_3\ t]\begin{bmatrix}X \cr Y \cr 0 \cr 1\end{bmatrix}$$

由于第三个坐标恒为 0，$r_3$ 被消掉：

$$\lambda x = K[r_1\ r_2\ t]\begin{bmatrix}X \cr Y \cr 1\end{bmatrix}$$

于是平面点到图像点之间只需要一个 $3 \times 3$ 矩阵：

$$\lambda x = H\begin{bmatrix}X \cr Y \cr 1\end{bmatrix}, \qquad H = K[r_1\ r_2\ t]$$

这个 $H$ 叫**单应矩阵**（Homography）。

> [!TIP]
> 这就是张氏标定法的入口：每张棋盘格图像都能估计一个 $H$；而每个 $H$ 又等于 $K[r_1\ r_2\ t]$。因为 $r_1,r_2$ 是同一个旋转矩阵的前两列，它们必须正交且等长，这些约束会反过来限制 $K$。

---

## 3.2 齐次坐标和单应矩阵

### 3.2.1 齐次坐标再看一眼

二维点 $(x,y)$ 的齐次形式是：

$$\tilde{x}=\begin{bmatrix}x \cr y \cr 1\end{bmatrix}$$

更一般地，$(x_1,x_2,x_3)^T$ 表示的普通坐标是：

$$(x,y)=\left(\frac{x_1}{x_3},\frac{x_2}{x_3}\right)$$

所以 $(x_1,x_2,x_3)^T$ 和 $k(x_1,x_2,x_3)^T$ 表示同一个点。这个“尺度不重要”的性质，是单应矩阵和相机矩阵都只定义到一个比例因子的原因。

### 3.2.2 单应矩阵是什么

单应矩阵 $H$ 描述一个平面到另一个平面的透视映射：

$$\tilde{x}' \sim H\tilde{x}$$

$H$ 是 $3 \times 3$ 非奇异矩阵，有 9 个元素，但只有 8 个自由度，因为整体乘一个非零常数不改变映射结果。

常见场景：

- 拍一本书封面：书的平面到照片平面是一个 $H$。
- 拍一张棋盘格：棋盘格平面到图像平面是一个 $H$。
- 相机纯旋转拍全景：两张图像之间也可以用 $H$ 描述。

### 3.2.3 四个点为什么能确定一个 H

一对平面点对应 $\tilde{x}_i \leftrightarrow \tilde{x}'_i$ 满足：

$$\tilde{x}'_i \sim H\tilde{x}_i$$

“方向相同”可以写成叉积为 0：

$$\tilde{x}'_i \times H\tilde{x}_i = 0$$

每对点提供两个独立线性方程。$H$ 有 8 个自由度，所以至少需要 4 对点。实际工程里通常用更多点，通过最小二乘和 RANSAC 抗噪声、抗误匹配。

> [!TIP]
> 四个点确定一个单应矩阵，前提是点配置不退化。比如四个点里有三个共线，约束就不够稳定。棋盘格有大量角点，所以比只拿四个角更稳。

---

## 3.3 用 DLT 求单应矩阵

DLT（Direct Linear Transformation）是求 $H$ 的标准线性方法。

给定 $N$ 对点 $(x_i,y_i)\leftrightarrow(u_i,v_i)$，构造线性方程：

$$Ah=0$$

其中 $h$ 是把 $H$ 展平成的 9 维向量。每对点给两行：

$$[-x_i,-y_i,-1,0,0,0,u_ix_i,u_iy_i,u_i]h=0$$

$$[0,0,0,-x_i,-y_i,-1,v_ix_i,v_iy_i,v_i]h=0$$

用 SVD 取最小奇异值对应的向量，reshape 成 $3 \times 3$，就得到 $H$。

```python
import numpy as np


def dlt_homography(src, dst):
    """
    Estimate H such that dst ~ H @ src.
    src, dst: arrays of shape (N, 2), N >= 4.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < 4:
        raise ValueError("At least 4 point pairs are required")

    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


src = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
])

dst = np.array([
    [120, 80],
    [580, 100],
    [560, 440],
    [100, 420],
])

H = dlt_homography(src, dst)
print(H)
```

OpenCV 中的工程入口是：

```python
H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
warped = cv2.warpPerspective(image, H, output_size)
```

> [!TIP]
> 原始 DLT 对坐标尺度敏感。实践中会先把点平移到质心为 0，并缩放到平均距离为 $\sqrt{2}$，在归一化坐标中求 $\tilde{H}$，最后再反归一化。这叫 normalized DLT，是数值稳定性的关键。

---

## 3.4 从单应矩阵到相机内参

现在进入张氏标定法的核心。

对于一张棋盘格图像：

$$H = K[r_1\ r_2\ t]$$

记 $H$ 的三列为：

$$H=[h_1\ h_2\ h_3]$$

因为 $K$ 可逆：

$$K^{-1}h_1 = \lambda r_1, \qquad K^{-1}h_2 = \lambda r_2, \qquad K^{-1}h_3 = \lambda t$$

其中 $\lambda$ 是未知尺度。关键约束来自旋转矩阵：

$$r_1^Tr_2=0, \qquad \|r_1\|=\|r_2\|$$

也就是说：

$$h_1^T K^{-T}K^{-1}h_2=0$$

$$h_1^T K^{-T}K^{-1}h_1=h_2^T K^{-T}K^{-1}h_2$$

令：

$$B=K^{-T}K^{-1}$$

每张棋盘格图片就给出两条关于 $B$ 的线性约束。$B$ 是对称矩阵，有 6 个元素，但整体尺度不重要，所以本质上有 5 个自由度。拍摄至少 3 张不同姿态的棋盘格，就能提供足够约束来求 $B$，进而恢复 $K$。

> [!TIP]
> 这就是“多拍几张不同角度棋盘格”的数学原因。每张图不是直接告诉你 $K$，而是先告诉你一个平面到图像的 $H$；多个 $H$ 共同约束同一个 $K$。

---

## 3.5 张氏标定法的完整流程

张氏标定法（Zhang's method）的主流程可以概括成五步：

1. 准备一个平面棋盘格，定义棋盘格坐标系，角点坐标为 $(X,Y,0)$。
2. 从多个角度拍摄棋盘格，检测每张图里的角点像素坐标。
3. 对每张图，用棋盘格平面点和图像点估计单应矩阵 $H_i$。
4. 利用 $H_i=K[r_1\ r_2\ t]$ 中 $r_1,r_2$ 的正交和等长约束，线性求出初始 $K$。
5. 以 $K$、每张图的 $R,t$、畸变参数为初值，最小化所有角点的重投影误差，做非线性优化。

OpenCV 的 `calibrateCamera` 封装了这些步骤：

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None
)
```

输出含义：

| 输出 | 含义 |
|------|------|
| `ret` | 平均重投影误差 |
| `K` | 内参矩阵 |
| `dist` | 畸变参数 |
| `rvecs/tvecs` | 每张棋盘格图的外参 |

> [!TIP]
> 张氏法的线性部分给出一个合理初值；最终结果通常来自非线性优化。优化目标很朴素：用当前参数把每个棋盘格角点投影回图像，和检测到的角点位置比较，让所有误差尽量小。

---

## 3.6 标定为什么需要“不同姿态”

如果所有棋盘格照片都几乎正对相机，得到的 $H$ 彼此太相似，对 $K$ 的约束会很弱。好的标定图应该覆盖：

- 棋盘格在图像中心和四角都出现过。
- 有正视、左倾、右倾、上倾、下倾。
- 棋盘格大小有变化，既有近一点也有远一点。
- 图像清晰，棋盘格平整，没有明显反光。

常见问题：

| 问题 | 可能原因 |
|------|---------|
| 重投影误差很大 | 角点检测不准、图片模糊、棋盘格不平 |
| 主点偏得离谱 | 棋盘格总在图像同一区域 |
| 畸变参数很奇怪 | 高阶参数过拟合，或角点存在系统误差 |
| 每次结果差很多 | 图片太少，姿态覆盖不足 |

> [!CAUTION]
> 标定时要固定焦距、固定对焦、固定分辨率。变焦、自动对焦或裁剪缩放都会改变 $K$ 的有效值。

---

## 3.7 图像矫正：单应矩阵的另一个直接应用

单应矩阵不只用于标定，也能做平面图像矫正。比如把一张斜拍的书封面拉成正视图。

```python
import cv2
import numpy as np

src = np.array([
    [120, 80],
    [580, 100],
    [560, 440],
    [100, 420],
], dtype=np.float32)

dst = np.array([
    [0, 0],
    [400, 0],
    [400, 300],
    [0, 300],
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src, dst)
rectified = cv2.warpPerspective(image, H, (400, 300))
```

这和标定里的棋盘格 $H$ 是同一种数学对象：都是平面到平面的透视映射。区别只是用途不同：

- 图像矫正：用 $H$ 把一个平面拉正。
- 张氏标定：用多个 $H$ 反推出 $K$。

---

## 3.8 本章小结、问题与练习

本章主线是：

```text
平面点 -> 单应矩阵 H -> H = K[r1 r2 t] -> 约束 K -> 标定初值
```

最重要的三句话：

1. 任意平面到图像之间可以用单应矩阵 $H$ 描述。
2. 棋盘格因为 $Z=0$，完整投影 $K[R|t]$ 会简化成 $H=K[r_1\ r_2\ t]$。
3. 张氏标定法利用 $r_1,r_2$ 的正交和等长约束，从多个 $H$ 中恢复 $K$。

### 苏格拉底时刻

1. 为什么棋盘格必须是平面？如果棋盘格翘起来，$H$ 的假设会怎样？
2. 一个 $H$ 有 8 个自由度，为什么 4 对点刚好够？
3. 为什么至少需要多张不同姿态的棋盘格图像，而不是一张正对相机的照片？
4. 张氏法线性求出的 $K$ 为什么还需要非线性优化 refine？

### 实操练习

**练习 1：四点求 H**

手动选一本书封面的四个角，用 `cv2.getPerspectiveTransform` 或 DLT 求 $H$，把封面拉成正视图。

**练习 2：验证 DLT**

用本章的 `dlt_homography` 估计 $H$，再用 `cv2.findHomography` 估计 $H$。比较二者把四个角点投影后的误差。

**练习 3：理解棋盘格姿态**

拍几张棋盘格照片，只观察角点分布，不跑标定。判断哪些照片对标定有帮助，哪些姿态太重复。用本章的约束解释你的判断。

### 延伸阅读

- 本书内：[[第 1 章 相机模型]] · [[第 2 章 坐标系转换]] · [[第 6 章 优化基础]]
- Hartley & Zisserman, *Multiple View Geometry*, Ch.2：齐次坐标与射影几何
- Hartley & Zisserman, Ch.4：DLT、归一化和鲁棒估计
- Zhang, “A Flexible New Technique for Camera Calibration”, IEEE TPAMI, 2000
- OpenCV Camera Calibration Tutorial: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
