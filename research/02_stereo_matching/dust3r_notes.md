# DUSt3R (CVPR 2024)

- Authors: Shuzhe Wang et al. (NAVER LABS)
- arXiv: 2312.14132

## Core idea
Cast stereo/multi-view 3D reconstruction as direct pointmap regression from image pairs. No camera calibration or pose priors needed.

## Architecture
- Standard Transformer encoder + decoder
- Input: arbitrary image collections
- Output: 3D pointmaps, depth maps, pixel matches, focal lengths, poses

## Key
- Unifies monocular and binocular reconstruction
- No calibration needed
- SOTA on monocular & multi-view depth, relative pose

## Follow-ups
- MV-DUSt3R+ (Dec 2024): multi-view from sparse views in ~2s
- MASt3R
- Awesome list: github.com/leo-frank/awesome-dust3R
