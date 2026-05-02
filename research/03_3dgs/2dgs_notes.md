# 2D Gaussian Splatting (arXiv:2403.17888)

- Authors: Binbin Huang et al. (ShanghaiTech)

## Core Idea
Replace 3D volumetric Gaussians with 2D oriented planar Gaussian disks.
Each disk = surface element with a normal direction.

## Key contributions
- Perspective-accurate 2D splat via ray-disk intersection (not projection approximation)
- Depth distortion + normal consistency regularizers
- SOTA geometry on DTU, T&T, Mip-NeRF360

## Benefit over 3DGS
- Much better surface reconstruction (3DGS struggles with geometry extraction)
- Naturally represents thin surfaces
