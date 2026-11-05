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

The deterministic methods all use the same 2D vision model: Faster R-CNN ResNet-50 FPN v2, COCO-pretrained (torchvision), with no training.

| # | Method | Type | Sensors | File |
|---|---|---|---|---|
| 1 | Stitched BEV + 2D detector (**baseline**) | Deterministic | Camera | `deterministic_bev_cam.py` |
| 2 | Per-camera detection, bottom-center projected to ground | Deterministic | Camera | `cam_ground_project.py` |
| 3 | Camera rays matched to LiDAR points (Isaiah) | Deterministic | Camera + LiDAR | `deterministic_BEV.py` (on `master`) |
| 4 | BEVFormer tiny | Model | Camera | `run_mini.py` |
| 5 | BEVFormer FP16 | Model | Camera | `run_bev_fp16.py` |
| 6 | BEVDet R50 | Model | Camera | `run_bevdet_r50.py` |
