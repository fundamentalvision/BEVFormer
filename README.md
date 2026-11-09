# Project 3: BEVFormer + nuScenes

## Running the eval benchmark

`eval_benchmark.py` scores every method's saved `outputs/<method>/results.json` and prints mAP, mATE and runtime. It does not run the methods, so run each method's script first (`outputs/` is git-ignored, so results are not shared through git).

```bash
cd DeterministicBev                    # the project folder
source .venv/bin/activate              # or `source activate.sh` on the BEVFormer machine

# 1. produce results (only needed when a method changes)
python deterministic_bev_cam.py        # stitched BEV baseline  -> outputs/cam_bev_detect/
python cam_ground_project.py           # ground projection      -> outputs/cam_ground_project/
python deterministic_bev_lidar.py      # camera + LiDAR         -> outputs/lidar_bev/

# 2. score everything
python eval_benchmark.py
```

Expected output, one line per method:

```
[ground_truth] mAP 1.0000 | mATE 0.000 m | no timing        <- sanity check
[cam_bev_detect] mAP ... | mATE ... m | ... s total, ... s/sample (cpu)
```

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
