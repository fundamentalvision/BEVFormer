import time
start = time.perf_counter()   # time the whole script, from the first line to saving results

from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-mini", dataroot="v1.0-mini", verbose=False)


import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from eval_benchmark import val_sample_tokens, ego_pose, ego_box_to_global, save_results, save_timing

# Idea: instead of warping the whole image onto the ground (where tall objects get smeared),
# run the detector on each raw camera image and only put ONE pixel per box on the ground:
# the bottom-center of the box, which is where the object touches the road.

CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

# ========== 2D DETECTOR (same as the BEV baseline) ==========
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

# COCO label id -> nuScenes class (only these 6 exist in both)
COCO_TO_NUSCENES = {1: "pedestrian", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

# get a mean size just for output format for evaluator, does not affect detection
# a 2D box has no real 3D size, so use each class's average size from mini_train (not val, to avoid leakage)
train_scenes = create_splits_scenes()["mini_train"]
sizes = {}
for ann in nusc.sample_annotation:
    scene = nusc.get("scene", nusc.get("sample", ann["sample_token"])["scene_token"])
    name = category_to_detection_name(ann["category_name"])
    if scene["name"] in train_scenes and name in COCO_TO_NUSCENES.values():
        sizes.setdefault(name, []).append(ann["size"])
mean_size = {name: np.mean(s, axis=0).tolist() for name, s in sizes.items()}   # [w, l, h]

# the same object can show up in two neighbouring cameras; boxes of the same class
# closer than this (m) are treated as one object and only the higher score is kept
MERGE_DIST = 2.0

# ========== PROCESSING ==========
results = {}
tokens = val_sample_tokens(nusc)
for i, token in enumerate(tokens):
    sample = nusc.get("sample", token)
    ego_boxes = []   # every box from all 6 cameras, in the ego frame

    for cam_name in CAMERAS:
        # read the calibration for this camera
        sd = nusc.get("sample_data", sample["data"][cam_name])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        # intrinsic matrix
        K_inv = np.linalg.inv(np.array(cs["camera_intrinsic"]))    # pixel -> ray direction (camera frame)
        R = Quaternion(cs["rotation"]).rotation_matrix             # camera frame -> ego frame rotation
        cam_pos = np.array(cs["translation"])                      # camera position in ego frame (z = height above ground)

        # run the detector on the raw camera image
        img = cv2.cvtColor(cv2.imread("v1.0-mini/" + sd["filename"]), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img).permute(2, 0, 1).float() / 255   # H x W x 3 in 0-255 -> 3 x H x W in 0-1
       # detection with vision model
        with torch.no_grad():
            detections = model([image])[0]

        for box, label, score in zip(detections["boxes"].tolist(), detections["labels"].tolist(), detections["scores"].tolist()):
            if label not in COCO_TO_NUSCENES:
                continue
            name = COCO_TO_NUSCENES[label]
            # top left, bottom right
            x1, y1, x2, y2 = box

            # bottom-center pixel of the box
            u, v = (x1 + x2) / 2, y2

            # shoot a ray from the camera through that pixel, expressed in the ego frame
            ray = R @ (K_inv @ np.array([u, v, 1.0]))   # (x fwd, y left, z up)
            if ray[2] >= 0:      # ray points level or up: it never hits the ground, skip
                continue

            # walk along the direction of the ray, find how long is it to reach to the pixel point
            s = -cam_pos[2] / ray[2]
            ground = cam_pos + s * ray                  

            # find the car's center 3D position, in the 3D world ego frame
            w, l, h = mean_size[name]
            direction = ray[:2] / np.linalg.norm(ray[:2])
            center_xy = ground[:2] + direction * (w + l) / 4 # estimate to be the center of the car

            ego_boxes.append({"center": [center_xy[0], center_xy[1], h / 2], "size": [w, l, h],
                              "yaw": 0.0, "name": name, "score": score})

    # merge duplicates across cameras: highest score first, drop any later box of the same class nearby
    ego_boxes.sort(key=lambda b: b["score"], reverse=True)
    kept = []
    for b in ego_boxes:
        duplicate = False
        for k in kept:
            dist = np.hypot(b["center"][0] - k["center"][0], b["center"][1] - k["center"][1])
            if k["name"] == b["name"] and dist < MERGE_DIST:
                duplicate = True
                break
        if not duplicate:
            kept.append(b)

    # package into correct format for benchmarking
    t, q = ego_pose(nusc, token)
    results[token] = [ego_box_to_global(b, t, q, token) for b in kept]

    print(f"{i + 1}/{len(tokens)}: {len(results[token])} boxes ({len(ego_boxes)} before merging)", flush=True)

elapsed = time.perf_counter() - start
save_timing(elapsed, len(tokens), "cpu", "outputs/cam_ground_project/timing.json")
print(f"Runtime: {elapsed:.1f} s total, {elapsed / len(tokens):.2f} s/sample")

save_results(results, "outputs/cam_ground_project/results.json")
print("Saved outputs/cam_ground_project/results.json")
