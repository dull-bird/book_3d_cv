# Monodepth2 (ICCV 2019)

- Authors: Godard et al. (UCL/Caltech/Niantic)
- arXiv: 1806.01260

## Key contributions
1. Per-pixel minimum reprojection loss: min over source views (handles occlusions)
2. Auto-masking stationary pixels: ignore unchanged pixels between frames
3. Full-resolution multi-scale sampling: upsample before computing loss

## Architecture
- Encoder: ResNet18 (11M params), ImageNet pretrained
- Decoder: U-Net with skip connections
- Pose Network: ResNet18, 6-channel input, predicts 6-DoF relative pose
- Loss: L1 + SSIM (α=0.85) + edge-aware smoothness

## KITTI Results
- Mono: AbsRel 0.115, δ<1.25 = 0.877
- Mono+Stereo: AbsRel 0.106
