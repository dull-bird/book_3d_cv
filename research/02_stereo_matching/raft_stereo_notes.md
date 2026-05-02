# RAFT-Stereo (3DV 2021 Best Paper)

- Authors: Lahav Lipson, Zachary Teed, Jia Deng (Princeton)
- arXiv: 2109.07547

## Architecture
1. Feature Encoder: left+right, 1/4 or 1/8 resolution, 256ch, instance norm
2. Context Encoder: left only, batch norm, init GRU hidden state
3. 3D Correlation Pyramid: HxWxW (1D search along epipolar line), 4-level 1D avg pool
4. Multi-Level GRU: 3 GRUs at 1/8, 1/16, 1/32 resolution, cross-connected

## Key innovation: Multi-Level GRU
- Cross-connected across resolutions
- Only highest-res GRU does correlation lookup and disparity update
- Slow-Fast variant: lower-res GRUs update more frequently, 52% speedup (0.132s→0.05s)

## Results
- Middlebury: #1, 4.74% bad-2px
- ETH3D: #1, 2.44 bad-1px
- KITTI-2015: #2 foreground
