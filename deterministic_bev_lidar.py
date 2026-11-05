from pathlib import Path
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name
import torch
import os
from dotenv import load_dotenv
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from eval_benchmark import val_sample_tokens, ego_pose, ego_box_to_global, save_results

ROOT = Path(__file__).resolve().parent
CAMERAS = set(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
COCO_TO_NUSCENES = {1: "pedestrian", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

load_dotenv()

class DeterministicBEV:
    def __init__(self):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=True)
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

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
            sd = self.nusc.get("sample_data", sample["data"][cam_name]) # sample data
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]) # calibrated sensor
        
            # fetching transformation ref frames
            K = np.array(cs["camera_intrinsic"]) # intrinsic lens matrix (already calibrated)
            #'samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg'
            image_path = os.path.join(os.getenv("NUSCENES_PATH"), sd["filename"])
            if os.path.exists(image_path):
                img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            else:
                print(f"Image not found: {image_path}")
                continue

            objects = self._detect_objects(img)

            rays = self.camera_uvs_to_rays(cam_name, objects["centroid"], scene_record)
            ego_locations = self.rays_to_ego_depth_via_lidar(rays, scene_record, cam_name)
            detections["labels"].extend(objects["label"])
            detections["ego_locations"].extend(ego_locations)
            detections["scores"].extend(objects["score"])
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
        for i, token in enumerate(tokens):
            scene_record = self.nusc.get("sample", token)
            w, l, h = mean_size[name]
            t, q = ego_pose(self.nusc, token)
            detections = self.process_frame(scene_record, token)
            x, y = detections["ego_locations"][i]
            score = detections["scores"][i]
            
            ego_box = {"center": [x, y, h / 2], "size": [w, l, h], "yaw": 0.0, "name": name, "score": score}
            results[token].append(ego_box_to_global(ego_box, t, q, token))
            results[token] = detections
        return results

    def _detect_objects(self, image):
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

    def _camera_to_world(self, camera, points, scene_record):
        cam_token = scene_record['data'][camera]
        cam_data = self.nusc.get('sample_data', cam_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        return points @ calibrated_sensor['rotation'].T + calibrated_sensor['translation']
    

    def _nearest_direction_points(rays, points, max_angle_deg=2.0):
        point_norms = np.linalg.norm(points, axis=1)
        point_directions = points / point_norms[:, None]

        scores = point_directions @ rays.T
        best_indices = np.argmax(scores, axis=0)
        best_scores = scores[best_indices, np.arange(len(rays))]

        matched = best_scores >= np.cos(np.deg2rad(max_angle_deg))

        selected_points = np.full_like(rays, np.nan)
        depths = np.full(len(rays), np.nan)

        selected_points[matched] = points[best_indices[matched]]
        depths[matched] = np.sum(selected_points[matched] * rays[matched], axis=1)

        return selected_points, depths

    # This function takes a list of rays and projects them into the lidar frame.
    def rays_to_ego_depth_via_lidar(self, rays, scene_record, ref_camera):
        lidar_token = scene_record['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token']).T

        pc, times = LidarPointCloud.from_file_multisweep(
            self.nusc, 
            sample_rec=scene_record, 
            chan='LIDAR_TOP', 
            ref_chan=ref_camera, # Automatically transforms into this camera's frame
            nsweeps=10.0
        )

        cast_points, _ = self._nearest_direction_points(rays, pc.points)
        world_points = self._camera_to_world(ref_camera, cast_points, scene_record)
        return world_points

if __name__ == "__main__":
    bev = DeterministicBEV()
    print("Evaluating...")
    print(bev.evaluate())