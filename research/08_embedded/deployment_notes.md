# 嵌入式部署调研

## 芯片
- 地瓜机器人 旭日5: 10 TOPS, 8核 A55, 3W, Ubuntu 22.04
- NVIDIA Jetson Orin AGX: 275 TOPS, cuVSLAM 3.8ms tracking
- Jetson Orin NX: 100 TOPS
- 华为 Atlas 200 DK: 12W

## 优化手段
1. 量化(FP16→INT8): PTQ first, QAT if needed
2. 剪枝(结构>非结构): 移除channel/head
3. 蒸馏: teacher→student压缩
4. 级联顺序: 蒸馏→剪枝→量化

## 实际部署例子
- DepthAnythingV2-Small + TensorRT FP16: Jetson Nano 可行
- cuVSLAM: Jetson Orin AGX, 3.8ms/camera, 6% GPU
- ZED SDK 5.0 TERRA AI: 2MP depth in 30ms on Orin Nano
