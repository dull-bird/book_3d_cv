# DROID-SLAM (NeurIPS 2021)

- Authors: Teed, Deng (Princeton)
- arXiv: 2108.10869

## Key
- Deep visual SLAM: differentiable BA layer
- GRU-based update operator predicts optical flow corrections
- DBA: Gauss-Newton with Schur complement for joint pose+depth optimization
- Monocular training, generalize to stereo/RGB-D at test time

## Results
- TartanAir: 62% error reduction (mono), 60% (stereo)
- ETH-3D: #1 AUC
- EuRoC: 82% error reduction among zero-failure methods
