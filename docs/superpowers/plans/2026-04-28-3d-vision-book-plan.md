# 《3D 视觉：从原理到实践》实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成开源书 MVP（基础篇 + 单目深度估计 + 双目立体匹配 + 3D Gaussian Splatting）的研究与写作，通过 mdBook 在线发布。

**Architecture:** 四阶段流水线 —— 基础设施搭建 → 逐模块"研究→写作"循环 → 全书整合 → 发布。每个模块严格先研究后写作，AI 基于一手文献生成初稿，作者技术审查。

**Tech Stack:** Markdown (写作), mdBook (构建在线阅读), GitHub (托管), Obsidian (知识管理与交叉引用)

---

## 文件结构

```
20_Projects/book_3d_cv/
├── README.md                       # 项目说明 + 参与指南
├── book.toml                       # mdBook 配置
├── src/
│   ├── SUMMARY.md                  # 全书目录
│   ├── 00_basics/                  # 基础篇：不变的脚手架
│   │   ├── 00_index.md             # 本篇导读
│   │   ├── 01_camera_model.md      # 相机模型
│   │   ├── 02_projection.md        # 投影几何
│   │   ├── 03_multi_view.md        # 多视图几何入门
│   │   ├── 04_depth_repr.md        # 深度表示
│   │   ├── 05_coordinates.md       # 坐标系转换
│   │   └── 06_optimization.md      # 优化基础
│   ├── 01_monocular_depth/         # 模块A：单目深度估计
│   │   ├── 00_index.md
│   │   ├── 01_intuition.md         # 第一阶：直觉
│   │   ├── 02_principles.md        # 第二阶：原理
│   │   └── 03_deployment.md        # 第三阶：部署
│   ├── 02_stereo_matching/         # 模块B：双目立体匹配
│   │   ├── 00_index.md
│   │   ├── 01_intuition.md
│   │   ├── 02_principles.md
│   │   └── 03_deployment.md
│   ├── 03_3dgs/                    # 模块C：3D Gaussian Splatting
│   │   ├── 00_index.md
│   │   ├── 01_intuition.md
│   │   ├── 02_principles.md
│   │   └── 03_deployment.md
│   └── appendix/
│       └── paper_index.md          # 全书论文索引
├── research/                       # 研究笔记（写作原材料）
│   ├── 00_basics/
│   │   └── reading_notes.md
│   ├── 01_monocular_depth/
│   │   ├── reading_notes.md
│   │   └── literature_map.md
│   ├── 02_stereo_matching/
│   │   ├── reading_notes.md
│   │   └── literature_map.md
│   └── 03_3dgs/
│       ├── reading_notes.md
│       └── literature_map.md
├── assets/                         # 图片、Mermaid 导出的图表
│   └── diagrams/
├── templates/
│   └── module_template.md          # 模块写作模板（供 AI 参考）
└── docs/superpowers/
    ├── specs/2026-04-28-3d-vision-book-design.md
    └── plans/2026-04-28-3d-vision-book-plan.md
```

---

## 阶段 0：基础设施搭建

### Task 0.1: 创建项目骨架

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p "20_Projects/book_3d_cv"/{src/{00_basics,01_monocular_depth,02_stereo_matching,03_3dgs,appendix},research/{00_basics,01_monocular_depth,02_stereo_matching,03_3dgs},assets/diagrams,templates}
```

- [ ] **Step 2: 初始化 Git**

```bash
cd "20_Projects/book_3d_cv"
git init
```

- [ ] **Step 3: 写入 .gitignore**

```
# .gitignore
.superpowers/
.DS_Store
```

```bash
echo '.superpowers/' > .gitignore
echo '.DS_Store' >> .gitignore
```

- [ ] **Step 4: 初始化 mdBook**

```bash
cd "20_Projects/book_3d_cv"
mdbook init --title "3D 视觉：从原理到实践" --ignore none
```

- [ ] **Commit**

```bash
git add -A
git commit -m "chore: initialize book project skeleton"
```

### Task 0.2: 创建模块写作模板

**Files:**
- Create: `20_Projects/book_3d_cv/templates/module_template.md`

- [ ] **Step 1: 写入模板文件**

写入 `templates/module_template.md`，内容为每个模块的三阶结构 + 章末元素的 Markdown 模板：

```markdown
# [模块名称]

## 模块导读

> 本模块预计阅读时间：X 分钟
> 前置知识：基础篇第 X、Y 节
> 读完本模块后，你可以：...

## 第一阶：直观理解

### 一个场景

[具体的痛点场景，让读者产生"我需要这个"的感觉]

### 核心直觉

[用类比建立直觉，不涉及数学推导]

### 技术全景

[Mermaid mindmap 展示该领域的技术分支]

### 十年关键突破

[Mermaid flowchart timeline 展示里程碑论文]

### Mini Case：上手跑一个

[用 5 分钟跑通一个现成模型，看到结果]

## 第二阶：原理解析

### 第一性原理：问题的本质是什么？

[剥离表象，回到物理/几何/信息论的基本事实]

### 长链推演

[从基本假设逐步推导到 SOTA 方法，每一步标注来源论文]

### 算法核心

[关键公式 + 人话翻译。公式引用原文出处]

### 方法演进对比

[Mermaid xychart 对比不同方法的性能演进]

### Code Lens

[核心代码段 + 逐行解释。代码来自官方仓库或亲手验证]

## 第三阶：部署实战

### 模型选型

[Mermaid quadrantChart：精度 × 速度四象限，标注各模型位置]

### 战争故事

[真实部署中踩过的坑，每个坑标注环境信息和解决方案]

### 数据陷阱

[训练集偏差、域迁移、评估指标——具体案例]

### 端到端案例

[从输入到输出的完整 pipeline，附代码]

## 苏格拉底时刻

1. [一个直击本质的反问]
2. [另一个]

## 关键论文清单

| 年份 | 论文 | 一句话贡献 |
|------|------|-----------|
| ... | ... | ... |

## 实操练习

1. [具体操作，含预期结果]
2. [一个会失败的场景，让读者思考为什么]

## 延伸阅读

- 本书内：[[模块X]]、[[模块Y]]
- 外部：[论文/教程链接]
```

- [ ] **Commit**

```bash
git add templates/
git commit -m "feat: add module writing template"
```

### Task 0.3: 编写 README

**Files:**
- Create: `20_Projects/book_3d_cv/README.md`

- [ ] **Step 1: 写入 README**

```markdown
# 3D 视觉：从原理到实践

一本面向所有人的 3D 视觉 AI 开源书。从直观理解到原理推导到部署实战，覆盖近 10 年重要学术成果。

## 阅读

在线版：https://[your-username].github.io/3d-vision-book

## 进度

| 模块 | 状态 |
|------|------|
| 基础篇：3D 视觉的脚手架 | 🔴 待开始 |
| 模块A：单目深度估计 | 🔴 待开始 |
| 模块B：双目立体匹配 | 🔴 待开始 |
| 模块C：3D Gaussian Splatting | 🔴 待开始 |

## 参与贡献

本书遵循"研究先行"原则：每个模块在写作前必须先完成一手文献研读。贡献请先开 Issue 讨论文献范围。

## 许可

[待定：建议 CC BY-NC-SA 4.0]
```

- [ ] **Commit**

```bash
git add README.md
git commit -m "docs: add README with project overview"
```

---

## 阶段 1：基础篇 —— 3D 视觉的脚手架

> **启动条件：** 作者指定并准备好基础篇的核心文献。
>
> **核心文献预设：**
> 1. Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2nd Ed.), Cambridge University Press, 2004. （奠基教材，英文原版）
> 2. Szeliski, *Computer Vision: Algorithms and Applications* (2nd Ed.), Springer, 2022. （参考教材）
> 3. 其他由作者根据各小节补充指定

### Task 1.1: 基础篇文献研读

**前置：** 作者提供上述核心文献的访问方式（PDF/纸质书）。

- [ ] **Step 1: 建立研究笔记文件**

```bash
touch research/00_basics/reading_notes.md
touch research/00_basics/literature_map.md
```

- [ ] **Step 2: 研读 Hartley & Zisserman 关键章节并做笔记**

每个子主题对应教材章节，产出阅读笔记。笔记格式：

```markdown
# 基础篇阅读笔记

## 相机模型 (Hartley & Zisserman Ch.6)

### 关键公式
- 投影矩阵: P = K[R|t] (公式 6.8, p.157)
- ...

### 核心 Insight
- ...

### 与人话的对应
- "内参矩阵 K" → 相机自己的属性，和外界无关
- ...
```

- [ ] **Step 3: 绘制基础篇文献地图**

`research/00_basics/literature_map.md`，用 Mermaid 展示教材章节与本书子主题的映射关系。

- [ ] **Step 4: 代码验证相机模型关键公式**

用 Python/NumPy 手动实现一次投影（针孔模型 → 畸变校正），确保公式理解正确。代码存入 `research/00_basics/code_verify.py`，输出注释。

- [ ] **Commit**

```bash
git add research/00_basics/
git commit -m "research: complete basics chapter literature review"
```

### Task 1.2: 基础篇导读 + 相机模型

**Files:**
- Create: `src/00_basics/00_index.md`
- Create: `src/00_basics/01_camera_model.md`

**前置：** Task 1.1 完成，研究笔记可用。

- [ ] **Step 1: 写导读（00_index.md）**

基于研究笔记，AI 生成基础篇导读。包含：本节在全书中的位置、学习目标、前置要求（无）、预计阅读时间。

- [ ] **Step 2: 写相机模型（01_camera_model.md）**

按三阶结构展开：
- 一阶：从"你的手机摄像头在做什么"切入，针孔相机类比小孔成像
- 二阶：投影矩阵 P=K[R|t] 的推导（来源 H&Z Ch.6），内参外参分别的含义，畸变模型
- 三阶：不同相机的内参标定实战（OpenCV calibrateCamera）
- 章末：苏格拉底时刻 + 论文清单 + 练习

**关键约束：每一个公式必须标注在 H&Z 中的出处（章节+页码）。**

- [ ] **Commit**

```bash
git add src/00_basics/
git commit -m "feat: draft basics - intro and camera model"
```

### Task 1.3: 投影几何

**Files:**
- Create: `src/00_basics/02_projection.md`

**前置：** Task 1.1 研究笔记。

- [ ] **Step 1: AI 基于研究笔记生成投影几何初稿**

按三阶结构：
- 一阶：跟读者说"3D 世界怎么变成 2D 照片"
- 二阶：齐次坐标的优雅之处（来源 H&Z Ch.2），透视投影矩阵
- 三阶：投影在实际代码中的表现（OpenCV projectPoints）
- 章末要素

- [ ] **Commit**

```bash
git add src/00_basics/02_projection.md
git commit -m "feat: draft basics - projection geometry"
```

### Task 1.4: 多视图几何入门

**Files:**
- Create: `src/00_basics/03_multi_view.md`

- [ ] **Step 1: AI 基于研究笔记生成多视图几何初稿**

重点是 H&Z 中关于对极几何、本质矩阵 E、基础矩阵 F、三角化（Ch.9-12）的内容。这是后来学习双目匹配的前置。

- [ ] **Commit**

```bash
git add src/00_basics/03_multi_view.md
git commit -m "feat: draft basics - multi-view geometry intro"
```

### Task 1.5: 深度表示 + 坐标系

**Files:**
- Create: `src/00_basics/04_depth_repr.md`
- Create: `src/00_basics/05_coordinates.md`

- [ ] **Step 1: 写深度表示**

视差图、深度图、点云——各自怎么来的、优缺点是啥、在什么场景用哪个。把三种表示放在同一张对比表里。

- [ ] **Step 2: 写坐标系转换**

世界→相机→像素坐标的完整转换链，一张总结性 Mermaid flowchart。

- [ ] **Commit**

```bash
git add src/00_basics/04_depth_repr.md src/00_basics/05_coordinates.md
git commit -m "feat: draft basics - depth representations and coordinate systems"
```

### Task 1.6: 优化基础 + 基础篇收尾

**Files:**
- Create: `src/00_basics/06_optimization.md`

- [ ] **Step 1: 写优化基础**

最小二乘的直觉、RANSAC 的优雅（内点/外点思想）、Bundle Adjustment 为什么是 3D 视觉的核心优化框架。

- [ ] **Step 2: 在 00_index.md 中补充 Mini Case**

"用手机拍两张照片，手动算桌子的距离"——一个具体的动手练习，引导读者用 OpenCV 跑一遍对极几何 → 三角化流程。代码放在正文中。

- [ ] **Commit**

```bash
git add src/00_basics/06_optimization.md src/00_basics/00_index.md
git commit -m "feat: draft basics - optimization and mini case"
```

### Task 1.7: 基础篇作者审查

- [ ] **Step 1: 作者通读基础篇全部内容，核查技术准确性**

重点核查：
- 公式推导是否正确（对照 H&Z 原文）
- 专业术语的中文翻译是否准确一致
- Mini Case 是否可实际跑通

- [ ] **Step 2: 修正并定稿基础篇 v0.1**

```bash
git add src/00_basics/
git commit -m "review: author review and polish of basics chapter v0.1"
```

---

## 阶段 2：模块 A —— 单目深度估计

> **启动条件：** 作者指定单目深度估计的核心文献。
>
> **核心文献预设：**
> 1. Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NeurIPS 2014. （开山之作）
> 2. Godard et al., "Unsupervised Monocular Depth Estimation with Left-Right Consistency", CVPR 2017. （自监督范式）
> 3. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer" (MiDaS), TPAMI 2022.
> 4. Yang et al., "Depth Anything V2", arXiv 2024.
> 5. Bochkovskii et al., "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", arXiv 2024.
> 6. 作者补充的其他文献

### Task 2.1: 模块 A 文献研读

- [ ] **Step 1: 建立研究笔记文件**

```bash
touch research/01_monocular_depth/reading_notes.md
touch research/01_monocular_depth/literature_map.md
```

- [ ] **Step 2: 研读并做笔记**

按时间线阅读 Eigen → Monodepth → MiDaS → Depth Anything → Depth Pro。每篇论文笔记包含：
- 解决的问题 + 前人方法的局限
- 核心创新（架构、Loss、训练策略）
- 关键公式 + 人话翻译
- 实验结论 + 局限

- [ ] **Step 3: 绘制文献演进地图**

`research/01_monocular_depth/literature_map.md`：一张 flowchart 展示从监督→自监督→Foundation Model 的范式转移，标注每篇论文的位置。

- [ ] **Step 4: 代码验证**

拉取 Depth Anything V2 官方代码，在自己的环境中跑通推理，记录环境配置和输出结果。存入 `research/01_monocular_depth/code_log.md`。

- [ ] **Commit**

```bash
git add research/01_monocular_depth/
git commit -m "research: monocular depth estimation literature review"
```

### Task 2.2: 模块 A — 第一阶（直觉）

**Files:**
- Create: `src/01_monocular_depth/00_index.md`
- Create: `src/01_monocular_depth/01_intuition.md`

- [ ] **Step 1: AI 基于研究笔记生成导读 + 第一阶初稿**

导读：模块概述、阅读时间、前置知识（引用基础篇具体章节）。

第一阶内容：
- 场景："你拍了一张风景照，想知道山离你多远。人看一眼就大概知道，这是怎么办到的？"
- 直觉：单目线索——透视、纹理梯度、相对大小、遮挡、光影
- Mindmap：单目深度估计的技术全景
- Timeline flowchart：2014 Eigen → 2017 Monodepth → 2022 MiDaS v3 → 2024 Depth Anything V2 / Depth Pro
- Mini Case：用 Depth Anything V2 对一张照片跑推理，展示输入输出

- [ ] **Commit**

```bash
git add src/01_monocular_depth/
git commit -m "feat: draft monocular depth - intro and intuition"
```

### Task 2.3: 模块 A — 第二阶（原理）

**Files:**
- Create: `src/01_monocular_depth/02_principles.md`

- [ ] **Step 1: AI 基于研究笔记生成原理初稿**

第一性原理：单张 2D 图像到 3D 深度——为什么是 ill-posed？本质是丢失了一个维度，必须靠先验补。

长链推演：
1. 监督学习阶段——Eigen 的多尺度网络为什么有效？尺度不变损失 (Scale-Invariant Loss) 的数学
2. 自监督阶段——Monodepth 的左-右一致性，光度损失，视差平滑
3. Foundation Model 阶段——MiDaS 的混合数据集训练，Depth Anything 的数据引擎
4. 最新——Depth Pro 的 metric depth 预测 + 焦距估计

Method 演进对比 xychart：横轴时间、纵轴 metric (δ<1.25)。

- [ ] **Commit**

```bash
git add src/01_monocular_depth/02_principles.md
git commit -m "feat: draft monocular depth - principles and derivations"
```

### Task 2.4: 模块 A — 第三阶（部署）

**Files:**
- Create: `src/01_monocular_depth/03_deployment.md`

- [ ] **Step 1: AI 基于研究笔记生成部署初稿**

- Quadrant chart：精度 × 推理速度（MiDaS small/large, Depth Anything S/B/L, Depth Pro, ZoeDepth 的位置）
- 战争故事：单目深度在移动端/嵌入式设备上的推理坑（ONNX 导出、量化精度损失）
- 数据陷阱：室内训练的模型放到室外、合成数据训练的模型放到真实场景——域迁移问题
- 端到端：从一张照片到 3D 点云，代码逐行注释
- 章末：苏格拉底时刻（"如果未来手机标配 ToF 传感器，单目深度估计的核心价值还会是'测距'吗？"）

- [ ] **Commit**

```bash
git add src/01_monocular_depth/03_deployment.md
git commit -m "feat: draft monocular depth - deployment and practice"
```

### Task 2.5: 模块 A 作者审查

- [ ] **Step 1: 作者通读模块 A 全部内容，核查技术准确性**

重点核查：
- 各论文的核心贡献描述是否准确
- Loss 公式推导是否正确
- Mini Case 和端到端案例的代码是否可复现

- [ ] **Step 2: 修正并定稿**

```bash
git add src/01_monocular_depth/
git commit -m "review: author review of monocular depth module v0.1"
```

---

## 阶段 3：模块 B —— 双目立体匹配

> **启动条件：** 作者指定双目立体匹配的核心文献。
>
> **核心文献预设：**
> 1. Scharstein & Szeliski, "A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms", IJCV 2002. （经典综述）
> 2. Mayer et al., "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation" (DispNet), CVPR 2016.
> 3. Chang & Chen, "Pyramid Stereo Matching Network" (PSMNet), CVPR 2018.
> 4. Lipson et al., "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching", 3DV 2021.
> 5. Wen et al., "FoundationStereo: Zero-Shot Stereo Matching", CVPR 2025.
> 6. Almeida da Silva et al., "Towards Understanding 3D Vision: the Role of Gaussian Curvature", arXiv 2025. （GC 视角）
> 7. 作者补充的其他文献

### Task 3.1: 模块 B 文献研读

- [ ] **Step 1: 建立研究笔记**

```bash
touch research/02_stereo_matching/reading_notes.md
touch research/02_stereo_matching/literature_map.md
```

- [ ] **Step 2: 研读并做笔记**

按时间线阅读：经典管线综述 → DispNet → PSMNet → RAFT-Stereo → FoundationStereo + GC 论文。特别关注：
- 经典四步管线（匹配代价 → 代价聚合 → 视差计算 → 精化）与 AI 方法的对应
- GC 论文中低曲率作为几何先验的论证和数据

- [ ] **Step 3: 绘制文献演进地图**

`research/02_stereo_matching/literature_map.md`

- [ ] **Step 4: 代码验证**

拉取 FoundationStereo 官方代码，跑通推理。记录环境配置和输出。如公开代码不可用，跑通 CREStereo 或 RAFT-Stereo 的官方推理。存入 `research/02_stereo_matching/code_log.md`。

- [ ] **Commit**

```bash
git add research/02_stereo_matching/
git commit -m "research: stereo matching literature review"
```

### Task 3.2: 模块 B — 第一阶（直觉）

**Files:**
- Create: `src/02_stereo_matching/00_index.md`
- Create: `src/02_stereo_matching/01_intuition.md`

- [ ] **Step 1: AI 基于研究笔记生成导读 + 第一阶初稿**

- 场景："你有两只眼睛，闭上左眼和闭上右眼看到的东西不一样。这个差异（视差）里藏着距离信息——你的大脑每天都在做双目匹配。"
- 直觉：通过左右图"找到同一点"来算距离，三角化。为什么难？——白墙没有纹理、树叶来回重复、玻璃会反射
- Mindmap 全景 + Timeline flowchart
- Mini Case：用 FoundationStereo 对 Middlebury 样本跑推理

- [ ] **Commit**

```bash
git add src/02_stereo_matching/
git commit -m "feat: draft stereo matching - intro and intuition"
```

### Task 3.3: 模块 B — 第二阶（原理）

**Files:**
- Create: `src/02_stereo_matching/02_principles.md`

- [ ] **Step 1: AI 基于研究笔记生成原理初稿**

- 第一性原理：双目匹配的本质是在另一张图中搜索对应点——这是一个对应问题 (correspondence problem)
- 经典管线：代价计算（Census Transform, NCC）→ 代价聚合（SGM）→ 视差计算（WTA）→ 视差精化
- 深度学习化：DispNet 把管线变成端到端 CNN → PSMNet 用 3D 卷积做 cost volume → RAFT-Stereo 借鉴光流的迭代优化 → FoundationStereo 引入 foundation model 做 zero-shot
- GC 视角专题：Gaussian Curvature 作为几何先验——低曲率意味着表面更可能是平面或柱面，SOTA 方法隐式学到了这个先验。你收件箱那篇论文的数据（LGC 指标、FoundationStereo 的 81.3% LGC）

- [ ] **Commit**

```bash
git add src/02_stereo_matching/02_principles.md
git commit -m "feat: draft stereo matching - principles and derivations"
```

### Task 3.4: 模块 B — 第三阶（部署）

**Files:**
- Create: `src/02_stereo_matching/03_deployment.md`

- [ ] **Step 1: AI 基于研究笔记生成部署初稿**

- Quadrant chart：精度 × 实时性（实时双目 vs 离线高精度）
- 部署陷阱：双目相机标定不准导致的系统性误差、弱光/低纹理场景的退化
- 端到端案例：从双目相机采集 → 矫正 → FoundationStereo 推理 → 点云生成，代码逐行
- 苏格拉底时刻："如果 FoundationStereo 已经做到了零样本泛化，大部分双目论文的研究范式会发生什么变化？"

- [ ] **Commit**

```bash
git add src/02_stereo_matching/03_deployment.md
git commit -m "feat: draft stereo matching - deployment and practice"
```

### Task 3.5: 模块 B 作者审查

- [ ] **Step 1: 作者通读核查**

重点核查：经典管线的描述准确性、GC 论文的数据引用是否正确、FoundationStereo 的技术描述。

- [ ] **Step 2: 修正并定稿**

```bash
git add src/02_stereo_matching/
git commit -m "review: author review of stereo matching module v0.1"
```

---

## 阶段 4：模块 C —— 3D Gaussian Splatting

> **启动条件：** 作者指定 3DGS 的核心文献。
>
> **核心文献预设：**
> 1. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020. （前身：NeRF）
> 2. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023. （开山之作）
> 3. 变体论文由作者根据兴趣指定（如 4DGS, Scaffold-GS, 2DGS, Mip-Splatting 等）
> 4. 作者补充的其他文献

### Task 4.1: 模块 C 文献研读

- [ ] **Step 1: 建立研究笔记**

```bash
touch research/03_3dgs/reading_notes.md
touch research/03_3dgs/literature_map.md
```

- [ ] **Step 2: 研读并做笔记**

先读 NeRF 理解"为什么需要新方法"（隐式表示 + 体渲染慢），再精读 3DGS（显式点云 + 可微光栅化 = 实时）。

关键概念笔记：3D Gaussian 的协方差矩阵、投影近似、球谐函数颜色、可微光栅化管线、自适应密度控制（克隆/分裂/致密化）。

- [ ] **Step 3: 绘制文献演进地图**

`research/03_3dgs/literature_map.md`：从 NeRF → 3DGS → 主流变体。

- [ ] **Step 4: 代码验证**

拉取 3DGS 官方代码，用自己的照片（20-30 张手机拍的小场景）跑通训练 → 渲染。记录环境配置、训练时间、显存占用、渲染效果。存入 `research/03_3dgs/code_log.md`。

- [ ] **Commit**

```bash
git add research/03_3dgs/
git commit -m "research: 3DGS literature review"
```

### Task 4.2: 模块 C — 第一阶（直觉）

**Files:**
- Create: `src/03_3dgs/00_index.md`
- Create: `src/03_3dgs/01_intuition.md`

- [ ] **Step 1: AI 基于研究笔记生成导读 + 第一阶初稿**

- 场景："你给朋友看了几张你家客厅的照片，他就能在脑海里'走进'这个房间，从任意角度看。3DGS 就是让计算机学会这个能力。"
- 直觉：在空间中撒一堆椭球体（Gaussian），每个有自己的位置、形状、颜色、透明度。从某个角度看过去，把它们叠起来就是画面。这和用像素拼图不同——它们是真正的 3D 东西。
- Mindmap 全景 + Timeline：NeRF (2020) → 3DGS (2023) → 爆炸式变体 (2024-)
- Mini Case：用预训练 3DGS 场景交互式旋转视角（如在浏览器中用 WebGL 查看）

- [ ] **Commit**

```bash
git add src/03_3dgs/
git commit -m "feat: draft 3DGS - intro and intuition"
```

### Task 4.3: 模块 C — 第二阶（原理）

**Files:**
- Create: `src/03_3dgs/02_principles.md`

- [ ] **Step 1: AI 基于研究笔记生成原理初稿**

- 第一性原理：场景表示的本质选择——用隐式函数（NeRF 的 MLP）还是显式几何（3DGS 的点云）？为什么显式+可微更好？
- 3D Gaussian 的数学：3D 协方差矩阵 Σ = RSSᵀRᵀ，投影到 2D 时的近似（EWA Splatting）
- 可微光栅化管线：frustum culling → 按深度排序 → 逐 tile 并行光栅化 → α-blending
- 训练过程：从 SfM 点云初始化 → 自适应密度控制 → 每 N 步克隆/分裂/致密化 → 移除低透明度和漂浮物
- SH 颜色：球谐函数 → view-dependent 颜色——为什么不同角度看同一地方颜色不一样？
- 变体速览：Scaffold-GS（用 anchor 减少冗余）、2DGS（用圆盘替代椭球处理表面）、4DGS（加时间维度）

- [ ] **Commit**

```bash
git add src/03_3dgs/02_principles.md
git commit -m "feat: draft 3DGS - principles and derivations"
```

### Task 4.4: 模块 C — 第三阶（部署）

**Files:**
- Create: `src/03_3dgs/03_deployment.md`

- [ ] **Step 1: AI 基于研究笔记生成部署初稿**

- Quadrant chart：渲染质量 × 训练+渲染速度（NeRF, 3DGS, Scaffold-GS, Mip-Splatting 的位置）
- 部署实战：用 COLMAP 做 SfM 获取初始点云 → 训练 3DGS → Web 端渲染（gsplat.js / VR 查看器）
- 常见问题：初始化不好怎么办（SfM 失败场景）、显存爆了怎么办、户外大场景怎么办
- 端到端案例：用手机拍 30 张照片 → COLMAP → 3DGS 训练 → Web 分享，完整 pipeline 代码
- 苏格拉底时刻："3DGS 用 100 万个 Gaussian 描述一个房间——这是稀疏还是密集？如果人类记忆一个房间只需要几个概念（'墙角'、'桌子'、'窗户'），3D 视觉领域离那种表征还有多远？"

- [ ] **Commit**

```bash
git add src/03_3dgs/03_deployment.md
git commit -m "feat: draft 3DGS - deployment and practice"
```

### Task 4.5: 模块 C 作者审查

- [ ] **Step 1: 作者通读核查**

重点核查：Gaussian 协方差矩阵的投影推导、SH 颜色描述、3DGS 训练流程的技术细节。

- [ ] **Step 2: 修正并定稿**

```bash
git add src/03_3dgs/
git commit -m "review: author review of 3DGS module v0.1"
```

---

## 阶段 5：全书整合与发布

### Task 5.1: 全书统稿

- [ ] **Step 1: 编写 SUMMARY.md（全书目录）**

```markdown
# 目录

[前言](README.md)

# 基础篇：3D 视觉的脚手架

- [导读](00_basics/00_index.md)
- [相机模型](00_basics/01_camera_model.md)
- [投影几何](00_basics/02_projection.md)
- [多视图几何入门](00_basics/03_multi_view.md)
- [深度表示](00_basics/04_depth_repr.md)
- [坐标系转换](00_basics/05_coordinates.md)
- [优化基础](00_basics/06_optimization.md)

# 主题模块

- [模块 A：单目深度估计](01_monocular_depth/00_index.md)
  - [直观理解](01_monocular_depth/01_intuition.md)
  - [原理解析](01_monocular_depth/02_principles.md)
  - [部署实战](01_monocular_depth/03_deployment.md)
- [模块 B：双目立体匹配](02_stereo_matching/00_index.md)
  - [直观理解](02_stereo_matching/01_intuition.md)
  - [原理解析](02_stereo_matching/02_principles.md)
  - [部署实战](02_stereo_matching/03_deployment.md)
- [模块 C：3D Gaussian Splatting](03_3dgs/00_index.md)
  - [直观理解](03_3dgs/01_intuition.md)
  - [原理解析](03_3dgs/02_principles.md)
  - [部署实战](03_3dgs/03_deployment.md)

# 附录

- [全书论文索引](appendix/paper_index.md)
```

```bash
git add src/SUMMARY.md
```

- [ ] **Step 2: 交叉引用检查**

遍历所有文件中的 `[[模块X]]` 和 `[文本](路径)` 链接，确保没有死链。

- [ ] **Step 3: 术语一致性检查**

基础篇和各模块中的中文术语保持统一。制作一份术语对照表放在附录中。

```bash
touch src/appendix/glossary.md
```

- [ ] **Commit**

```bash
git add src/SUMMARY.md src/appendix/
git commit -m "feat: finalize book structure - SUMMARY, cross-refs, glossary"
```

### Task 5.2: mdBook 构建与测试

- [ ] **Step 1: 配置 book.toml**

```bash
cd "20_Projects/book_3d_cv"
```

修改 `book.toml`：

```toml
[book]
title = "3D 视觉：从原理到实践"
authors = ["戴智文"]
language = "zh"
multilingual = false
src = "src"

[output.html]
mathjax-support = true
git-repository-url = "https://github.com/[username]/3d-vision-book"
edit-url-template = "https://github.com/[username]/3d-vision-book/edit/main/{path}"

[output.html.search]
enable = true
```

- [ ] **Step 2: 本地构建测试**

```bash
mdbook build
```

检查输出中是否有破链、渲染错误。

- [ ] **Step 3: 修复构建问题**

如有错误，修正后重新 build。

- [ ] **Commit**

```bash
git add book.toml
git commit -m "chore: mdBook configuration and build verification"
```

### Task 5.3: GitHub Pages 部署

- [ ] **Step 1: 创建 GitHub 仓库**

用户在 GitHub 上创建仓库 `3d-vision-book`（或自行命名）。

- [ ] **Step 2: 推送代码**

```bash
git remote add origin git@github.com:[username]/3d-vision-book.git
git branch -M main
git push -u origin main
```

- [ ] **Step 3: 配置 GitHub Actions 自动部署**

创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy mdBook
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: peaceiris/actions-mdbook@v1
        with:
          mdbook-version: 'latest'
      - run: mdbook build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./book
```

- [ ] **Step 4: 验证在线版本**

推送后确认 `https://[username].github.io/3d-vision-book` 可访问。

- [ ] **Commit**

```bash
git add .github/
git commit -m "ci: add GitHub Actions for mdBook deploy"
git push
```

### Task 5.4: 社交媒体发布

- [ ] **Step 1: 撰写知乎文章（第一篇——项目发布）**

基于 README 改写，介绍项目动机、MVP 范围、阅读链接。包含关键图表（全书架构图）。

- [ ] **Step 2: 发布并更新 README**

在 README 中添加知乎文章链接和在线书链接。

```bash
git add README.md
git commit -m "docs: add social media links to README"
git push
```

---

## 后续迭代列表（v0.2+）

以下模块在 MVP 完成后，按作者兴趣和优先级逐一启动（每个模块同样走"研究→写作→审查"流程）：

- 深度补全（Depth Completion）
- NeRF（作为 3DGS 的前身可先写，也可作为独立模块）
- VGGT
- SLAM
- SfM
- 多视图立体匹配 (MVS)
- 点云处理与分析

每次新增模块：新开 research/ 子目录 → 文献研读 → 在 src/ 下创建新文件夹 → 按模板写三阶 → 更新 SUMMARY.md → 发布新版本。

---

*计划完成。下一步：作者确认后，选择执行模式开始实施。*
