# COLMAP (CVPR 2016 + ECCV 2016)

- Authors: Schönberger, Frahm (ETH Zurich)
- SfM paper: CVPR 2016 "Structure-from-Motion Revisited"
- MVS paper: ECCV 2016 "Pixelwise View Selection for Unstructured MVS"

## SfM Pipeline
1. Correspondence Search: SIFT → exhaustive pairwise matching → geometric verification (F/E/H)
2. Incremental Reconstruction: init best pair → register new images (PnP + RANSAC) → triangulate → BA → iterate

## Key innovations
- Multi-model geometric verification (classify pairs as general/panoramic/planar)
- Multi-scale next-best-view selection
- RANSAC-based robust triangulation
- Iterative BA + filtering + re-triangulation
- Redundant view clustering (group BA for dense scenes)

## MVS
- Dense reconstruction after SfM
- Depth map estimation → fusion → meshing
