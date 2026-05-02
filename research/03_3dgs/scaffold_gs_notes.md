# Scaffold-GS (CVPR 2024 Highlight)

- Authors: Tao Lu et al. (CUHK MMLab)
- Project: city-super.github.io/scaffold-gs

## Core Idea
Instead of free-floating Gaussians, use anchor points to structure them hierarchically.
Anchors spawn k neural Gaussians with learnable offsets.
Gaussian attributes predicted by MLPs conditioned on anchor features + view direction + distance.

## Key
- Anchor growing/pruning via gradient and opacity significance
- Comparable quality to 3DGS, ~100 FPS, smaller storage
- More robust to transparent/specular/reflective regions
