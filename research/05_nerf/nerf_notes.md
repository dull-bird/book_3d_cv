# NeRF (ECCV 2020)

- Authors: Mildenhall et al. (UC Berkeley / Google)
- arXiv: 2003.08934

## Architecture
- Input: 5D (x,y,z,θ,φ) → positional encoding (L=10 for pos, L=4 for dir)
- MLP: 8 layers × 256ch ReLU (density branch) → 1 layer × 128ch (color branch)
- σ depends only on position; color depends on position + view direction

## Volume Rendering
- C(r) = ∫ T(t) σ(r(t)) c(r(t),d) dt
- Stratified sampling + coarse-to-fine hierarchical sampling

## Results
- SOTA on synthetic 360°, real forward-facing scenes
- Training: ~1-2 days per scene on single GPU
