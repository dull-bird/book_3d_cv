# Depth Anything V2 (NeurIPS 2024)

- Authors: Lihe Yang et al. (HKU/TikTok)
- arXiv: 2406.09414

## Architecture
- DINOv2 ViT encoder (S/B/L/G: 25M/97M/335M/1.3B params) + DPT decoder
- V2 change: use intermediate features from last 4 layers (stages 9-12)

## 3-Stage Training
1. Teacher on 595K synthetic images (5 datasets: Hypersim, VK2, BlendedMVS, IRS, TartanAir)
   - No real labels at all
2. Pseudo-label 62M unlabeled real images (8 sources: BDD100K, IN-21K, Places365, SA-1B...)
3. Student distillation on pseudo-labels only, no synthetic data

## Losses
- SSI Loss: affine-invariant L1 in log space, per-sample scale-shift alignment
- Gradient Matching Loss L_gm: L1 on spatial gradients, 2x weight vs SSI
  - Critical for sharp edges when using synthetic data
- Feature Alignment Loss: encourages student to preserve DINOv2 semantics
- Top-10% loss masking to handle pseudo-label noise

## Key insight
Replacing noisy real labels with pixel-perfect synthetic GT was the #1 improvement from V1.
