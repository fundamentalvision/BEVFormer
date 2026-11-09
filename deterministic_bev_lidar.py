import time
start = time.perf_counter()   # time the whole script, from the first line to saving results

from pathlib import Path
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name
from shapely import points
import torch
import os
import tqdm
from dotenv import load_dotenv
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from eval_benchmark import val_sample_tokens, ego_pose, ego_box_to_global, save_results, save_timing
from eval_benchmark import evaluate as evaluate_results

ROOT = Path(__file__).resolve().parent
CAMERAS = set(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
COCO_TO_NUSCENES = {1: "pedestrian", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

# documentation figures for one busy sample (the mini_val sample with the most annotated objects)
SAVE_FIGS = True
FIG_MIN_SCORE = 0.5   # only draw confident boxes so the figure stays readable (all boxes are still used for results)
import matplotlib.pyplot as plt

load_dotenv()

class DeterministicBEV:
    def __init__(self):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=True)
        self.device = torch.device("cpu")
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(self.device).eval()
        self.fig_token = None    # the sample to draw figures for (set in evaluate)
        self.fig_images = {}     # camera name -> image with boxes drawn on it
        self.fig_rays = []       # (camera position, LiDAR hit), ego frame, for the top-down plot

    # Returns a list of rays IN THE CAMERA FRAME, from pixel coordinates.
    def camera_uvs_to_rays(self, camera, input_uv, scene_record):
        cam_token = scene_record['data'][camera]
        cam_data = self.nusc.get('sample_data', cam_token)
        calibrated_sensor = self.nusc.get(
            'calibrated_sensor',
            cam_data['calibrated_sensor_token'],
        )

        intrinsics = np.asarray(calibrated_sensor['camera_intrinsic'], dtype=float)
        pixels = np.column_stack([
            np.asarray(input_uv, dtype=float),
            np.ones(len(input_uv)),
        ])
        rays = pixels @ np.linalg.inv(intrinsics).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

        return rays

    def process_frame(self, scene_record, token):
        detections = {
            "labels": [],
            "ego_locations": [],
            "scores": []
        }
        sample = self.nusc.get("sample", token)
        for cam_name in CAMERAS:
            sd = self.nusc.get("sample_data", sample["data"][cam_name])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        
            # fetching transformation ref frames
            K = np.array(cs["camera_intrinsic"]) # intrinsic lens matrix (already calibrated)
            #'samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg'
            image_path = os.path.join(self.nusc.dataroot, sd["filename"])
            if os.path.exists(image_path):
                img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            else:
                print(f"Image not found: {image_path}")
                continue
            objects = self._detect_objects(img)
            if not objects["label"]:
                continue

            rays = self.camera_uvs_to_rays(cam_name, objects["centroid"], scene_record)
            ego_locations = self.rays_to_ego_depth_via_lidar(rays, scene_record, cam_name)

            # documentation figure: 2D box, the centroid pixel the ray goes through, and the LiDAR distance
            if SAVE_FIGS and token == self.fig_token:
                for box, name, centroid, location, score in zip(objects["box"], objects["label"], objects["centroid"], ego_locations, objects["score"]):
                    if score < FIG_MIN_SCORE:
                        continue
                    x1, y1, x2, y2 = [int(c) for c in box]
                    if np.isfinite(location).all():   # LiDAR point found along the ray: green, with distance
                        color, text = (0, 255, 0), f"{name} {np.hypot(location[0], location[1]):.1f} m"
                        self.fig_rays.append((np.array(cs["translation"][:2]), location[:2]))
                    else:                             # no LiDAR point within 2 deg: red, dropped
                        color, text = (255, 0, 0), f"{name} no LiDAR hit"
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(img, (int(centroid[0]), int(centroid[1])), 10, (255, 255, 0), -1)
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                self.fig_images[cam_name] = img

            for name, location, score in zip(objects["label"], ego_locations, objects["score"]):
                if not np.isfinite(location).all():
                    continue # skip detections that don't have a valid location

                detections["labels"].append(name)
                detections["ego_locations"].append(location)
                detections["scores"].append(score)
        return detections

    def evaluate(self):
        results = {}
        tokens = val_sample_tokens(self.nusc)

        train_scenes = create_splits_scenes()["mini_train"]
        sizes = {}
        for ann in self.nusc.sample_annotation:
            scene = self.nusc.get("scene", self.nusc.get("sample", ann["sample_token"])["scene_token"])
            name = category_to_detection_name(ann["category_name"])
            if scene["name"] in train_scenes and name in COCO_TO_NUSCENES.values():
                sizes.setdefault(name, []).append(ann["size"])
        
        mean_size = {name: np.mean(s, axis=0).tolist() for name, s in sizes.items()}   # [w, l, h]
        self.fig_token = max(tokens, key=lambda tok: len(self.nusc.get("sample", tok)["anns"]))
        for token in tqdm.tqdm(tokens):
            # Load relevant data from the NuScenes sample
            sample = self.nusc.get("sample", token)
            detections = self.process_frame(sample, token)
            t, q = ego_pose(self.nusc, token)

            results[token] = []

            # Convert detections to global coordinates and package them for the evaluation function
            for name, location, score in zip(detections["labels"], detections["ego_locations"], detections["scores"]):
                x, y, z = location
                w, l, h = mean_size[name]
                ego_box = {"center": [float(x), float(y), float(z)], "size": [w, l, h], "yaw": 0.0, "name": name, "score": float(score)}
                results[token].append(ego_box_to_global(ego_box, t, q, token))

            if SAVE_FIGS and token == self.fig_token:
                self._save_figures(sample, detections, mean_size)

        return results

    def _save_figures(self, sample, detections, mean_size):
        os.makedirs(ROOT / "figs", exist_ok=True)

        # figure 1: the 6 camera images (fixed order) with boxes, ray pixels (yellow) and LiDAR distances
        order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        fig, axes = plt.subplots(2, 3, figsize=(18, 7))
        for ax, cam in zip(axes.flat, order):
            if cam in self.fig_images:
                ax.imshow(self.fig_images[cam])
            ax.set_title(cam)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(ROOT / "figs/lidar_bev_cameras.png", dpi=120)

        # figure 2: top-down view in the ego frame (car at the origin, forward = up, left = left)
        # plot coordinates are (-y, x) so that forward points up the page
        plt.figure(figsize=(8, 8))

        # one LiDAR sweep in the background: sensor frame -> ego frame
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_sd["filename"]))
        pts = pc.points[:3].T @ Quaternion(lidar_cs["rotation"]).rotation_matrix.T + np.array(lidar_cs["translation"])
        plt.scatter(-pts[:, 1], pts[:, 0], s=0.2, color="lightgray")

        for cam_xy, hit_xy in self.fig_rays:   # each camera ray, from the camera to the LiDAR point it matched
            plt.plot([-cam_xy[1], -hit_xy[1]], [cam_xy[0], hit_xy[0]], color="tab:orange", lw=0.6)
            plt.plot(-hit_xy[1], hit_xy[0], ".", color="tab:red")
        for name, location, score in zip(detections["labels"], detections["ego_locations"], detections["scores"]):
            if score < FIG_MIN_SCORE:   # our predicted boxes: centered on the LiDAR point, class mean size, yaw 0
                continue
            w, l, h = mean_size[name]
            plt.gca().add_patch(plt.Rectangle((-location[1] - w / 2, location[0] - l / 2), w, l, fill=False, color="tab:blue"))

        t, q = ego_pose(self.nusc, sample["token"])
        for ann_token in sample["anns"]:   # ground-truth boxes, moved from the global frame to the ego frame
            if self.nusc.get("sample_annotation", ann_token)["num_lidar_pts"] == 0:
                continue
            gt = self.nusc.get_box(ann_token)
            if category_to_detection_name(gt.name) not in COCO_TO_NUSCENES.values():
                continue
            gt.translate(-t)
            gt.rotate(q.inverse)
            corners = gt.bottom_corners()[:2]   # 2 x 4 footprint (x, y)
            plt.plot(list(-corners[1]) + [-corners[1][0]], list(corners[0]) + [corners[0][0]], color="tab:green")

        plt.plot(0, 0, "k^", markersize=10)
        plt.plot([], [], ".", color="lightgray", label="LiDAR points (1 sweep)")
        plt.plot([], [], color="tab:orange", label="camera ray")
        plt.plot([], [], ".", color="tab:red", label="matched LiDAR point")
        plt.plot([], [], color="tab:blue", label="prediction (mean size, yaw 0)")
        plt.plot([], [], color="tab:green", label="ground truth")
        plt.plot([], [], "k^", label="our car")
        plt.legend(loc="upper right")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.gca().set_aspect("equal")
        plt.xlabel("left / right (m)")
        plt.ylabel("back / forward (m)")
        plt.title("Camera + LiDAR, top-down (ego frame)")
        plt.tight_layout()
        plt.savefig(ROOT / "figs/lidar_bev_bev.png", dpi=120)
        plt.close("all")
        print("Saved figs/lidar_bev_cameras.png and figs/lidar_bev_bev.png")

    def _detect_objects(self, image):
        image = (torch.from_numpy(image).permute(2, 0, 1).float().to(self.device) / 255)    
        with torch.no_grad():        
            detections = self.model([image])[0]

        box_label_scores = {
            "box": [],
            "label": [],
            "centroid": [],
            "score": []
        }

        for box, label, score in zip(detections["boxes"].tolist(), detections["labels"].tolist(), detections["scores"].tolist()):
                if label not in COCO_TO_NUSCENES:
                    continue
                name = COCO_TO_NUSCENES[label]

                box_label_scores["box"].append(box)
                box_label_scores["label"].append(name)
                box_label_scores["score"].append(score)
                box_label_scores["centroid"].append(((box[2] + box[0]) / 2, (box[3] + box[1]) / 2))

        return box_label_scores

    def _camera_to_ego(self, camera, points, scene_record):
        cam_token = scene_record['data'][camera]
        cam_data = self.nusc.get('sample_data', cam_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        R = Quaternion(calibrated_sensor['rotation']).rotation_matrix
        camera_ego = points @ R.T + np.asarray(calibrated_sensor["translation"])

        camera_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        camera_rotation = Quaternion(camera_pose["rotation"]).rotation_matrix
        global_points = (camera_ego @ camera_rotation.T + np.asarray(camera_pose["translation"]))

        lidar_translation, lidar_rotation = ego_pose(self.nusc, scene_record["token"])
        return (global_points - lidar_translation) @ lidar_rotation.rotation_matrix
    
    @staticmethod
    def _nearest_direction_points(rays, points, max_angle_deg=2.0):
        points = points[np.isfinite(points).all(axis=1)]
        points = points[np.linalg.norm(points, axis=1) > 1e-6]

        if len(points) == 0 or len(rays) == 0:
            return (np.full((len(rays), 3), np.nan), np.full(len(rays), np.nan))
        point_norms = np.linalg.norm(points, axis=1)
        point_directions = points / point_norms[:, None]

        scores = point_directions @ rays.T
        best_indices = np.argmax(scores, axis=0)
        best_scores = scores[best_indices, np.arange(len(rays))]

        matched = best_scores >= np.cos(np.deg2rad(max_angle_deg))

        # Select the lidar points closest to the rays
        selected_points = np.full_like(rays, np.nan)
        depths = np.full(len(rays), np.nan)

        selected_points[matched] = points[best_indices[matched]]
        depths[matched] = np.sum(selected_points[matched] * rays[matched], axis=1)

        return selected_points, depths

    # This function takes a list of rays and projects them into the lidar frame.
    def rays_to_ego_depth_via_lidar(self, rays, scene_record, ref_camera):
        lidar_token = scene_record['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)

        pc, times = LidarPointCloud.from_file_multisweep(
            self.nusc, 
            sample_rec=scene_record, 
            chan='LIDAR_TOP', 
            ref_chan=ref_camera, # Automatically transforms into this camera's frame
            nsweeps=10
        )

        cast_points, _ = self._nearest_direction_points(rays, pc.points[:3].T)
        ego_points = self._camera_to_ego(ref_camera, cast_points, scene_record)
        return ego_points

if __name__ == "__main__":
    bev = DeterministicBEV()
    results = bev.evaluate()

    elapsed = time.perf_counter() - start
    save_timing(elapsed, len(results), "cpu", str(ROOT / "outputs/lidar_bev/timing.json"))
    print(f"Runtime: {elapsed:.1f} s total, {elapsed / len(results):.2f} s/sample")

    path = str(ROOT / "outputs/lidar_bev/results.json")
    save_results(results, path)
    evaluate_results(bev.nusc, "lidar_bev", path)