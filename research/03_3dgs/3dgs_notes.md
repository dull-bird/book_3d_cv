# 3D Gaussian Splatting (SIGGRAPH 2023)

- Authors: Kerbl, Kopanas, Leimkühler, Drettakis (Inria)
- arXiv: 2308.04079

## Pipeline
SfM sparse points → init 3D Gaussians → optimize + adaptive density control → tile-based rasterizer

## 3D Gaussian Representation
Each Gaussian: position μ, covariance Σ (scale s + quaternion rotation q), opacity α, SH coefficients (view-dependent color)

## Adaptive Density Control
- Clone: under-reconstructed regions (small gradient)
- Split: over-reconstructed regions (large gradient + large Gaussian)

## Tile-Based Rasterizer
1. Screen → 16x16 tiles, frustum culling, GPU Radix Sort by depth+tileID
2. Per-tile CUDA thread block, α-blend front-to-back, stop at α≈1
3. Backward: reuse sorted arrays, reconstruct intermediate opacity from stored α

## Performance
- Training: ~41min (Mip-NeRF360, 30K iters)
- Rendering: 134 fps @1080p
- Memory: 734MB
