# Literature Map: H&Z Chapters to Book Sections

```mermaid
graph TB
    subgraph HZ["H&Z Source Chapters"]
        HZ2["Ch.2-3<br/>Projective Geometry<br/>平面射影几何 & 3D 射影几何"]
        HZ6["Ch.6<br/>Camera Models<br/>相机模型"]
        HZ4["Ch.4<br/>Estimation<br/>估计理论"]
        HZ9["Ch.9<br/>Epipolar Geometry<br/>对极几何"]
        HZ10["Ch.10<br/>3D Reconstruction<br/>3D 重建"]
        HZ18["Ch.18<br/>Bundle Adjustment<br/>光束平差"]
    end

    subgraph Book["book_3d_cv: 3D 视觉：从原理到实践 — 基础篇"]
        B1["相机模型<br/>P = K[R|t]"]
        B2["投影几何<br/>齐次坐标、H矩阵、分层变换"]
        B3["多视图几何<br/>F矩阵、E矩阵、对极约束"]
        B4["深度表示<br/>三角测量、深度图、视差"]
        B5["坐标系转换<br/>世界→相机→图像→像素"]
        B6["优化基础<br/>DLT、RANSAC、BA"]
    end

    subgraph Supplements["补充来源 (非 H&Z)"]
        S1["深度学习相关<br/>NeRF, 3DGS, MVS 综述"]
        S2["传感器模型<br/>LiDAR, 深度相机, 双目"]
        S3["工程实践<br/>COLMAP, OpenCV API"]
    end

    %% Main mappings
    HZ6 -->|"P = K[R|t] 公式<br/>内外参分解、RQ分解<br/>仿射相机层级"| B1
    HZ6 -->|"世界→相机坐标变换<br/>K(图像→像素)变换"| B5

    HZ2 -->|"齐次坐标优雅性<br/>射影变换群<br/>l_∞, 圆环点<br/>度量性质恢复"| B2
    HZ2 -->|"分层坐标框架<br/>射影/仿射/度量/欧几里得"| B5

    HZ9 -->|"对极几何直觉<br/>F 矩阵定义与性质<br/>E 矩阵定义<br/>F ↔ K 关系"| B3

    HZ10 -->|"三角测量原理<br/>射影重建定理<br/>分层重建策略<br/>重建歧义性"| B4

    HZ4 -->|"DLT 算法<br/>代价函数层级<br/>MLE, Sampson 误差"| B6

    HZ18 -->|"BA 数学原理<br/>因子分解法<br/>稀疏求解"| B6

    S1 -->|"深度学习的 NVS/重建"| B4
    S2 -->|"主动传感器测距原理"| B4
    S3 -->|"代码级实现参考"| B1
    S3 -->|"代码级实现参考"| B3

    %% Cross-references
    HZ2 -.->|"2D 射影几何为<br/>3D 推广打基础"| HZ10
    HZ9 -.->|"F 矩阵是<br/>射影重建的输入"| HZ10
    HZ4 -.->|"估计方法是<br/>F/H/P 的计算工具"| HZ9
    HZ18 -.->|"BA 优化<br/>射影重建结果"| HZ10

    style HZ2 fill:#e1f5fe
    style HZ6 fill:#e1f5fe
    style HZ4 fill:#fff3e0
    style HZ9 fill:#e8f5e9
    style HZ10 fill:#e8f5e9
    style HZ18 fill:#fff3e0
    style B1 fill:#f3e5f5
    style B2 fill:#f3e5f5
    style B3 fill:#fce4ec
    style B4 fill:#fce4ec
    style B5 fill:#f3e5f5
    style B6 fill:#fce4ec
    style S1 fill:#eceff1
    style S2 fill:#eceff1
    style S3 fill:#eceff1
```

## Key Source Mapping Detail

### 相机模型 (`book_3d_cv` Section 1)

| 知识点 | H&Z Reference | Pages |
|--------|--------------|-------|
| Pinhole projection formula | section 6.1 | 154-155 |
| Calibration matrix K | section 6.1 | 155-157 |
| External params R, t | section 6.1 | 155-156 |
| General projective camera | section 6.2 | 158-163 |
| RQ decomposition to recover K, R | section 6.2.4 | 163-165 |
| Affine camera hierarchy | section 6.3 | 166-173 |
| Depth of points | section 6.2.3 | 162-163 |

### 投影几何 (`book_3d_cv` Section 2)

| 知识点 | H&Z Reference | Pages |
|--------|--------------|-------|
| Homogeneous coordinates | section 2.2.1 | 26-28 |
| Ideal points, line at infinity | section 2.2.2 | 28-30 |
| Projective transformation H | section 2.3 | 32-36 |
| Hierarchy of transformations | section 2.4 | 37-44 |
| Recovery of affine/metric properties | section 2.7 | 47-58 |
| Circular points, C*_inf | section 2.7.3 | 52-53 |
| Angle measurement in projective frame | section 2.7.4 | 54-55 |

### 多视图几何 (`book_3d_cv` Section 3)

| 知识点 | H&Z Reference | Pages |
|--------|--------------|-------|
| Epipolar geometry intuition | section 9.1 | 239-241 |
| Fundamental matrix F definition | section 9.2 | 241-246 |
| F properties (rank 2, 7 DOF) | section 9.2.4 | 245-246 |
| Essential matrix E | section 9.6 | 257-258 |
| F vs E relationship | section 9.6 | 257 |
| Camera matrices from F | section 9.5 | 254-256 |
| Special motions | section 9.3 | 247-250 |

### 深度表示 (`book_3d_cv` Section 4)

| 知识点 | H&Z Reference | Pages |
|--------|--------------|-------|
| Triangulation principle | section 10.1 | 262-264 |
| Projective reconstruction theorem | section 10.3 | 266-267 |
| Stratified reconstruction | section 10.4 | 267-273 |
| Reconstruction ambiguity | section 10.2 | 264-266 |

### 坐标系转换 (`book_3d_cv` Section 5)

| 知识点 | H&Z Reference | Pages |
|--------|--------------|-------|
| World to camera (R, t) | section 6.1 | 155-156 |
| Camera to image (K) | section 6.1 | 154-157 |
| Hierarchical frames | section 2.4, 2.7 | 37-44, 47-58 |

### 优化基础 (`book_3d_cv` Section 6)

| 知识点 | H&Z Reference | Pages |
|--------|--------------|-------|
| DLT algorithm | section 4.1 | 88-91 |
| Cost functions (algebraic, geometric, Sampson) | section 4.2 | 91-96 |
| Maximum Likelihood estimation | section 4.3 | 102-108 |
| RANSAC | section 4.7 | 116-123 |
| Bundle adjustment principle | section 18.1 | 434-436 |
| Factorization algorithm | section 18.2 | 436-440 |

## Dependency Graph

```mermaid
graph LR
    A["Ch.2: 2D Projective<br/>Geometry"] --> C["Ch.4: Estimation<br/>(DLT, RANSAC)"]
    A --> B["Ch.3: 3D Projective<br/>Geometry"]
    B --> D["Ch.6: Camera<br/>Models"]
    D --> E["Ch.9: Epipolar<br/>Geometry"]
    C --> E
    E --> F["Ch.10: 3D<br/>Reconstruction"]
    D --> F
    B --> F
    F --> G["Ch.18: Bundle<br/>Adjustment"]
    C --> G
```

*Last updated: 2026-04-28*
