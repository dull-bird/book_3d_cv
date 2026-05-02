# MVSNet (ECCV 2018)

- Authors: Yao Yao et al. (HKUST)
- arXiv: 1804.02505

## Pipeline
1. 2D CNN feature extraction (8-layer) on all views
2. Differentiable homography warping (plane sweep) → 3D cost volume
3. Variance-based cost metric: maps N-view features to single cost
4. Multi-scale 3D CNN regularization → softmax → probability volume
5. Soft argmin depth regression + refinement network

## Key innovations
- Variance cost metric: arbitrary N-view input
- Differentiable homography: end-to-end training
- Soft argmin: differentiable depth inference

## Results
- DTU: 0.527mm completeness (SOTA at time)
- Tanks and Temples: #1 (score 43.48)
