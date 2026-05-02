# MoGe (arXiv:2410.19115, CVPR 2025 Oral)

- Authors: Wang et al. (USTC + Microsoft Research)

## Core innovation: Affine-Invariant 3D Point Map
- Predict 3D point map P(HxWx3) instead of depth (HxW)
- Affine-invariant: P̂ ≈ sP + t (unknown global scale and shift)
- Eliminates ambiguity in depth supervision

## ROE Solver (Robust, Optimal, Efficient)
- Finds optimal (s,t) during training via parallel search
- Fast (~3ms) closed-form optimization

## Training Loss
- L1 loss weighted by inverse GT depth: L_G = Σ(1/z_i)·||s*·p̂_i + t* − p_i||₁
- Multi-scale local geometry loss: spherical region sampling

## Camera Recovery
- From affine-invariant point map → focal length + shift via optimization (~3ms)

## Backbone
- DINOv2 encoder

## Training Data
- ~9M frames, synthetic + real mixture

## Results
- >35% reduction in average relative point error on zero-shot benchmarks
