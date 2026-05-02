# ORB-SLAM3 (2020, T-RO)

- Authors: Campos et al. (Univ. Zaragoza)
- arXiv: 2007.11898

## Architecture: 3 threads
1. Tracking: ORB feature matching + motion-only BA, decides keyframes
2. Local Mapping: sliding window BA over local keyframes
3. Loop Closing: DBoW2 place recognition → pose graph optimization → full BA

## Key innovations
- Atlas multi-map: disconnected maps, merge on loop
- IMU integration: visual-inertial mode
- Improved place recognition (co-visible verification)

## Performance
- EuRoC: 3.6cm (stereo-inertial)
- TUM-VI: 9mm
