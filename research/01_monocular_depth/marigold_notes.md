# Marigold (CVPR 2024 Oral, Best Paper Candidate)

- Authors: Ke et al. (ETH Zurich)
- arXiv: 2312.02145 (Dec 2023)

## Core idea
Fine-tune Stable Diffusion v2 into affine-invariant monocular depth estimator.
Stable Diffusion's rich visual priors (trained on billions of images) generalize much better than training from scratch.

## Training
- Freeze SD VAE + most UNet layers
- Only fine-tune small part of UNet
- Use only synthetic data (no real depth)
- Train on single GPU, few days

## Inference
- DDIM scheduler, 1-4 steps
- Affine-invariant depth output

## Results
- >20% improvement over Depth Anything V1 on zero-shot benchmarks
- 10x+ slower than Depth Anything V2

## Follow-up: Marigold v2
- Extended to surface normals, intrinsic image decomposition
- LCM scheduler for 1-step inference
