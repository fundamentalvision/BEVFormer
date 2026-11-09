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

# ========== FIGURES FOR DOCUMENTATION ==========
# draw the method's steps for one busy sample: the mini_val sample with the most annotated objects
SAVE_FIGS = True
FIG_MIN_SCORE = 0.5   # only draw confident boxes so the figure stays readable (all boxes are still used for results)
import os
import matplotlib.pyplot as plt
fig_token = max(tokens, key=lambda tok: len(nusc.get("sample", tok)["anns"]))
fig_images = []   # (camera name, image with boxes drawn on it)
fig_rays = []     # (camera position, ground contact point), ego frame, for the top-down plot

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

            # documentation figure: 2D box, ground contact pixel, and estimated distance from the car
            if SAVE_FIGS and token == fig_token and score >= FIG_MIN_SCORE:
                dist = np.hypot(center_xy[0], center_xy[1])
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.circle(img, (int(u), int(v)), 10, (255, 0, 0), -1)
                cv2.putText(img, f"{name} {dist:.1f} m", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                fig_rays.append((cam_pos[:2], ground[:2]))

        if SAVE_FIGS and token == fig_token:
            fig_images.append((cam_name, img))

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

    if SAVE_FIGS and token == fig_token:
        os.makedirs("figs", exist_ok=True)

        # figure 1: the 6 camera images with boxes, contact points (red) and distances
        fig, axes = plt.subplots(2, 3, figsize=(18, 7))
        for ax, (cam, im) in zip(axes.flat, fig_images):
            ax.imshow(im)
            ax.set_title(cam)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("figs/ground_project_cameras.png", dpi=120)

        # figure 2: top-down view in the ego frame (car at the origin, forward = up, left = left)
        # plot coordinates are (-y, x) so that forward points up the page
        plt.figure(figsize=(8, 8))
        for cam_xy, ground_xy in fig_rays:   # each ray on the ground, from the camera to the contact point
            plt.plot([-cam_xy[1], -ground_xy[1]], [cam_xy[0], ground_xy[0]], color="lightgray", lw=0.8)
            plt.plot(-ground_xy[1], ground_xy[0], ".", color="tab:red")
        for b in kept:   # our predicted boxes: class mean size, yaw 0
            if b["score"] < FIG_MIN_SCORE:
                continue
            w, l, h = b["size"]
            cx, cy = b["center"][0], b["center"][1]
            plt.gca().add_patch(plt.Rectangle((-cy - w / 2, cx - l / 2), w, l, fill=False, color="tab:blue"))
        for ann_token in sample["anns"]:   # ground-truth boxes, moved from the global frame to the ego frame
            if nusc.get("sample_annotation", ann_token)["num_lidar_pts"] == 0:
                continue
            gt = nusc.get_box(ann_token)
            if category_to_detection_name(gt.name) not in COCO_TO_NUSCENES.values():
                continue
            gt.translate(-t)
            gt.rotate(q.inverse)
            corners = gt.bottom_corners()[:2]   # 2 x 4 footprint (x, y)
            plt.plot(list(-corners[1]) + [-corners[1][0]], list(corners[0]) + [corners[0][0]], color="tab:green")
        plt.plot(0, 0, "k^", markersize=10)
        plt.plot([], [], color="lightgray", label="camera ray")
        plt.plot([], [], ".", color="tab:red", label="ground contact point")
        plt.plot([], [], color="tab:blue", label="prediction (mean size, yaw 0)")
        plt.plot([], [], color="tab:green", label="ground truth")
        plt.plot([], [], "k^", label="our car")
        plt.legend(loc="upper right")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.gca().set_aspect("equal")
        plt.xlabel("left / right (m)")
        plt.ylabel("back / forward (m)")
        plt.title("Ground projection, top-down (ego frame)")
        plt.tight_layout()
        plt.savefig("figs/ground_project_bev.png", dpi=120)
        plt.close("all")
        print("Saved figs/ground_project_cameras.png and figs/ground_project_bev.png")

    print(f"{i + 1}/{len(tokens)}: {len(results[token])} boxes ({len(ego_boxes)} before merging)", flush=True)

elapsed = time.perf_counter() - start
save_timing(elapsed, len(tokens), "cpu", "outputs/cam_ground_project/timing.json")
print(f"Runtime: {elapsed:.1f} s total, {elapsed / len(tokens):.2f} s/sample")

save_results(results, "outputs/cam_ground_project/results.json")
print("Saved outputs/cam_ground_project/results.json")
