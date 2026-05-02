# CREStereo (CVPR 2022 Oral)

- Authors: Megvii Research + Tencent + UESTC
- arXiv: 2203.11483

## Key contributions
1. Adaptive Group Correlation Layer (AGCL)
   - 2D-1D alternate local search for non-ideal rectification
   - Deformable search window with learned offsets
   - LoFTR-style local feature attention in first cascade
   - Group-wise correlation (from GwcNet)

2. Cascaded Recurrent Network
   - 3 resolution levels: 1/16, 1/8, 1/4
   - Recurrent Update Module (RUM) at each level
   - All RUMs share weights

3. New synthetic dataset (~400GB Blender)
   - Thin structures, non-texture, metallic, transparent objects
   - ShapeNet + Blender trees

## Results
- Middlebury 2014: #1, Bad 2.0 = 3.71 (vs RAFT-Stereo 4.74)
- ETH3D: #1, Bad 1.0 = 0.98 (59.84% improvement)
