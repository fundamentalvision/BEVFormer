import time
start = time.perf_counter()   # time the whole script, from the first line to saving results

from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-mini", dataroot="v1.0-mini", verbose=False)
# print(len(nusc.sample))   # expect 404


import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from eval_benchmark import val_sample_tokens, ego_pose, ego_box_to_global, save_results, save_timing

# build the BEV grid in ego coords in meters
extent, res = 50.0, 0.1                   # cover +-50 m, 10 cm per pixel
size = int(2 * extent / res)              # 1000 x 1000 pixels

cols, rows = np.meshgrid(np.arange(size), np.arange(size))
# arrays to correspond the pixel coords in BEV image with the ego-frame coords in meters
# following same convention as nuscence dataset does (left is positive)
grid_x = extent - (rows + 0.5) * res      # forward distance (m); top row = +50 m ahead
grid_y = extent - (cols + 0.5) * res      # left distance (m); left column = +50 m to the left

# print(size, grid_x[0, 0], grid_y[0, 0], grid_x[-1, -1], grid_y[-1, -1])

# flatten the ground grid into points, same for every camera (assuming all are grounds for now)
pts = np.stack([grid_x.ravel(), grid_y.ravel(),
                np.zeros(grid_x.size), np.ones(grid_x.size)])   # 4 x N : (x, y, z=0, 1), last one just for easier matrix multiplication

# ========== PROCESSING ==============
# take each camera sensor view sample raw image into the BEV coordinate system
def camera_to_bev(sample, cam_name):
    # read the calibration for this camera
    sd = nusc.get("sample_data", sample["data"][cam_name]) # sample data
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]) # calibrated sensor

    # fetching transformation ref frames
    K = np.array(cs["camera_intrinsic"]) # intrinsic lens matrix (already calibrated)
    T_ego_cam = transform_matrix(cs["translation"], Quaternion(cs["rotation"]))   # camera's pose in ego frame
    T_cam_ego = np.linalg.inv(T_ego_cam)                                          # ego's frame in the cam frame

    # print("K =\n", K)
    # print("camera position in ego frame (x fwd, y left, z up):", cs["translation"])

    img = cv2.cvtColor(cv2.imread("v1.0-mini/" + sd["filename"]), cv2.COLOR_BGR2RGB)

    # show othe raw image
    # plt.imshow(img)
    # plt.title(cam_name)
    # plt.axis("off")
    # plt.show()

    # project the ground points into this camera
    cam = (T_cam_ego @ pts)[:3]               # the same points in the camera frame (drops 4th)
    uvw = K @ cam                             # apply the lens matrix
    depth = uvw[2]                            # distance in front of the camera
    u = uvw[0] / np.maximum(depth, 1e-6)      # pixel column in the image
    v = uvw[1] / np.maximum(depth, 1e-6)      # pixel row in the image

    h, w = img.shape[:2]
    valid = (depth > 0.1) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)

    valid = valid.reshape(size, size)
    # print("fraction of BEV seen by", cam_name, valid.mean())
    # print("front half:", valid[:500].mean(), " back half:", valid[500:].mean())

    # Sample the image colors into the BEV
    map_x = np.where(valid, u.reshape(size, size), -1).astype(np.float32)   # image column for each BEV pixel
    map_y = np.where(valid, v.reshape(size, size), -1).astype(np.float32)   # image row for each BEV pixel
    bev = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return bev, valid


# run every camera and stitch into one BEV
CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

def render_bev(sample):
    canvas = np.zeros((size, size, 3), np.uint8)   # the final BEV, starts all black
    filled = np.zeros((size, size), bool)          # which BEV pixels already have a color

    for cam_name in CAMERAS:
        bev, valid = camera_to_bev(sample, cam_name)
        new = valid & ~filled          # pixels this camera sees that nobody has filled yet (first camera wins overlaps)
        canvas[new] = bev[new]         # copy only those
        filled |= valid                # mark them as filled
    return canvas


# ========== 2D DETECTOR ON THE BEV (null baseline) ==========
# standard COCO-pretrained Faster R-CNN, the same detector every variant will use
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

# COCO label id -> nuScenes class (only these 6 exist in both)
COCO_TO_NUSCENES = {1: "pedestrian", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

# a BEV box has no real 3D size, so use each class's average size from mini_train (not val, to avoid leakage)
train_scenes = create_splits_scenes()["mini_train"]
sizes = {}
for ann in nusc.sample_annotation:
    scene = nusc.get("scene", nusc.get("sample", ann["sample_token"])["scene_token"])
    name = category_to_detection_name(ann["category_name"])
    if scene["name"] in train_scenes and name in COCO_TO_NUSCENES.values():
        sizes.setdefault(name, []).append(ann["size"])
mean_size = {name: np.mean(s, axis=0).tolist() for name, s in sizes.items()}   # [w, l, h]

results = {}
tokens = val_sample_tokens(nusc)
for i, token in enumerate(tokens):
    sample = nusc.get("sample", token) # a specific moment in time captured
    canvas = render_bev(sample)

    # run the detector on the BEV image
    image = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255   # H x W x 3 in 0-255 -> 3 x H x W in 0-1
    with torch.no_grad():
        detections = model([image])[0]

    # package into correct format for benchmarking
    t, q = ego_pose(nusc, token)
    results[token] = []
    for box, label, score in zip(detections["boxes"].tolist(), detections["labels"].tolist(), detections["scores"].tolist()):
        if label not in COCO_TO_NUSCENES:
            continue
        name = COCO_TO_NUSCENES[label]
        x1, y1, x2, y2 = box

        # box center in BEV pixels -> ego meters, same mapping as grid_x and grid_y
        x = extent - (y1 + y2) / 2 * res
        y = extent - (x1 + x2) / 2 * res
        w, l, h = mean_size[name]
        ego_box = {"center": [x, y, h / 2], "size": [w, l, h], "yaw": 0.0, "name": name, "score": score}
        results[token].append(ego_box_to_global(ego_box, t, q, token))

    print(f"{i + 1}/{len(tokens)}: {len(results[token])} boxes", flush=True)

elapsed = time.perf_counter() - start
save_timing(elapsed, len(tokens), "cpu", "outputs/cam_bev_detect/timing.json")
print(f"Runtime: {elapsed:.1f} s total, {elapsed / len(tokens):.2f} s/sample")

save_results(results, "outputs/cam_bev_detect/results.json")
print("Saved outputs/cam_bev_detect/results.json")

plt.imshow(canvas)
plt.title("6 cameras -> BEV (last sample)")
plt.axis("off")
plt.show()
