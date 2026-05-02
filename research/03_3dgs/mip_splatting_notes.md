# Mip-Splatting (CVPR 2024 Oral, Best Student Paper)

- Authors: Zehao Yu et al. (Tübingen)
- arXiv: 2311.16493

## Problem
3DGS produces aliasing artifacts when zooming or changing focal length.
Gaussians smaller than a pixel cause high-frequency noise.

## Solution: Two filters
1. 3D smoothing filter: constrains Gaussian size based on max sampling frequency
   from training views → eliminates zoom-in artifacts
2. 2D Mip filter: replaces 2D dilation operation with box filter simulation →
   mitigates aliasing when changing camera distance/focal length
