# Project 3: BEVFormer + nuScenes

## Environment

Already installed: Conda `bevformer`, Python 3.11, PyTorch 2.8 / CUDA 12.8.

```bash
cd "/home/isa/Documents/Comp Robo/Project3"
source activate.sh
```

## Start

```bash
python run_mini.py
```

Evaluates all 81 mini validation frames using the shared six-class mAP/mATE evaluator.
Detections and evaluation results: `outputs/bevformer_tiny/`.

All models and `eval_benchmark.py` share this activation. Checkpoints: `ckpts/`.
[Shared requirements](requirements.txt).

---

# BEV Cam (Jack)

This section is for the camera-only deterministic BEV (`deterministic_bev_cam.py`). It runs on CPU and does not need the BEVFormer environment above.

## Setup (macOS, one time)

Run from the project folder. Python 3.12 is used because `nuscenes-devkit` may not have wheels for newer Python versions.

```bash
cd /Users/welchj/Desktop/DeterministicBev
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install nuscenes-devkit pyquaternion opencv-python matplotlib
```

## Run

Activate the venv in each new terminal, then run from the project folder (the dataset path `v1.0-mini` is relative):

```bash
cd /Users/welchj/Desktop/DeterministicBev
source .venv/bin/activate
python deterministic_bev_cam.py
```

Expected output: `404`, then `(900, 1600, 3)` for each of the 6 cameras.

## Plan: comparing 6 BEV methods

All methods are scored the same way: nuScenes `v1.0-mini` val (81 samples), mAP and mATE over 6 classes (car, truck, bus, pedestrian, bicycle, motorcycle), using `eval_benchmark.py`.

- **mAP** (mean Average Precision): did the method find the objects without false alarms? 0–1, higher is better. A prediction counts if its center is within 0.5 / 1 / 2 / 4 m of a real object; AP is averaged over those thresholds.
- **mATE** (mean Average Translation Error): for correct matches, how far off is the center, in meters? Lower is better. If nothing matches, nuScenes reports 1.0 as a placeholder.

The deterministic methods all use the same 2D vision model: Faster R-CNN ResNet-50 FPN v2, COCO-pretrained (torchvision), with no training.

| # | Method | Type | Sensors | File |
|---|---|---|---|---|
| 1 | Stitched BEV + 2D detector (**baseline**) | Deterministic | Camera | `deterministic_bev_cam.py` |
| 2 | Per-camera detection, bottom-center projected to ground | Deterministic | Camera | `cam_ground_project.py` |
| 3 | Camera rays matched to LiDAR points (Isaiah) | Deterministic | Camera + LiDAR | `deterministic_BEV.py` (on `master`) |
| 4 | BEVFormer tiny | Model | Camera | `run_mini.py` |
| 5 | BEVFormer FP16 | Model | Camera | `run_bev_fp16.py` |
| 6 | BEVDet R50 | Model | Camera | `run_bevdet_r50.py` |

---

# Robot Localization
Author: Jack, Isa

For Olin ENGR3590 Computational Introduction to Robotics

This README is a comprehensive report that contains both conceptual and programmatic information regarding implementation details of our machine vision project.

## 1. Project Overview

The goal of this project is to learn the vision perception layers of modern autonomous vehicles (AV) by examining industrial practices and implementing our own algorithms. With a growing popularity of autonomous vehiclces companies like Waymo, Zoox, Tesla, etc., we want to gain a comprehensive understanding of what and why certain sensors and techinques are used, as well as the performance of different methods. This write up will present the implementation, results, and learning from comparing 6 different approaches we used, and a comparison analysis using the benchmark we picked.

## 2. Dataset and setup

To keep the project reasonably scoped, we decided to use an existing dataset instead of collecting data ourselves. This way, we also have well-processed truth to reference to when benchmarking at the end.

In terms of the database choice, we picked nuScenes, a driving dataset recorded in Boston and Singapore. We specifically use `v1.0-mini` dataset, consisting of 10 scenes (404 samples) out of the full 1,000. It is small enough to run the deterministic methods on a laptop CPU. Each scene is a 20-second drive with 3D boxes annotated on keyframes ("samples"). We chose it for the following reasons:

- **Full 360° multi-angle camera:** There are six cameras covering every direction around the car, which is a common practice on autonomous vehicles as well as what a bird's-eye view needs.
- **Multi-source sensor data:** Sensor data from camera, LiDAR and radar are all provided on the same frames. All sensors are recorded together, with calibration already and the car's position for every frame. This lets us compare sensor setups (camera only vs camera + LiDAR) on exactly the same scenes.
- **Compatibility with pretrained models:** BEVFormer and BEVDet publish checkpoints trained on nuScenes, so we can evaluate the learned methods without training them ourselves.


## 3. Evaluation
- mAP and mATE (center-distance thresholds).
- Runtime: whole-script time, device noted (CPU vs GPU caveat).
- Ground-truth sanity check (mAP 1.0, mATE 0.0).

## 4. Methods
### 4.1 Deterministic (shared detector: Faster R-CNN R50-FPN v2, COCO)
- Stitched BEV + detector (baseline)
- Per-camera detection + ground projection
- Camera rays + LiDAR (Isaiah)

### 4.2 Learned
- BEVFormer FP32 / FP16
- BEVDet R50

## 5. Results
- Main table: mAP, mATE, runtime, device for all 6 methods.
- Per-class AP table (0.5 / 1 / 2 / 4 m).
- Optional figure: BEV predictions vs ground truth.

## 6. Discussion
- Why the stitched BEV fails (IPM smears tall objects).
- Ground projection: detection works, depth is the bottleneck (flat-ground triangle, error grows with distance).
- Accuracy vs compute (6 detector passes vs 1).
- Camera vs LiDAR: measured vs inferred distance.
- Deterministic vs learned.

## 7. Limitations
- Small eval set (81 samples).
- No yaw / size / velocity estimation in the deterministic methods.
- CPU vs GPU timing not directly comparable.
- Generic COCO detector, not trained on driving data.

## 8. Future work
- Camera + radar late fusion.
- Tracking across frames.
- Combining depth cues (ground contact + known size).

## 9. Conclusion

## References
- nuScenes, Faster R-CNN, BEVFormer, BEVDet.
