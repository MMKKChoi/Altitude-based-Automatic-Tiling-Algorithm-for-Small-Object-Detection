# Altitude-based Automatic Tiling Algorithm for Small Object Detection

![wens](https://github.com/MMKKChoi/Altitude-based-Automatic-Tiling-Algorithm-for-Small-Object-Detection/assets/125550210/37ae9d87-23f6-4169-acd0-237b0c21158c)

+ This work is output of master degree in WENS lab (https://wens.re.kr), Dept. of IT Convergence Engineering, Kumoh National Institute of Technology

## Introduction
+ In this paper, a method is proposed to perform consistent detection of small objects across all altitudes using UAVs. This approach determines the number of tiles for detecting small objects at each altitude. Real-time object detection was conducted on the AI board mounted on the UAV, and for this purpose, the model was optimized using a lightweighting process.

<img width="808" alt="tt" src="https://github.com/MMKKChoi/Altitude-based-Automatic-Tiling-Algorithm-for-Small-Object-Detection/assets/125550210/287d287c-e8d4-40f4-81c0-1e2938905014">
<br><br>
+ The tiling algorithm for image processing to find small objects
<br><br><br><br>
<p align="center">
  <img width="635" alt="ss" src="https://github.com/MMKKChoi/Altitude-based-Automatic-Tiling-Algorithm-for-Small-Object-Detection/assets/125550210/93d39e33-28cd-4b1a-b0f8-1431e4b16f76">
</p>

+ Comparison of object detection results: The first row presents the ground truth with yellow boxes, the second row presents results using YOLOv5, the third row presents results with 2x2 tiling, and the fourth row presents results using the proposed method.

## Environment
+ Jetson Xavier/Orin NX (Jetpack 5.1.2)
+ Yolov5 (https://github.com/ultralytics/yolov5)


## Installation

## Paper
+ M. K. Choi and S. Y. Shin, "Altitude-Based Automatic Tiling Algorithm for Small Object Detection," The Journal of Korean Institute of Communications and Information Sciences, vol. 49, no. 3, pp. 424-434, 2024. DOI: 10.7840/kics.2024.49.3.424.
