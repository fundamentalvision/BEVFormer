# Copyright 2021 Toyota Research Institute.  All rights reserved.
#import functools
from collections import OrderedDict

import numpy as np
import seaborn as sns
from torch.utils.data import Dataset
from tqdm import tqdm

#from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import BoxMode
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

#from tridet.data import collect_dataset_dicts
from projects.mmdet3d_plugin.dd3d.structures.boxes3d import GenericBoxes3D
from projects.mmdet3d_plugin.dd3d.structures.pose import Pose
from projects.mmdet3d_plugin.dd3d.utils.geometry import project_points3d
from projects.mmdet3d_plugin.dd3d.utils.visualization import float_to_uint8_color

#  https://github.com/nutonomy/nuscenes-devkit/blob/9b209638ef3dee6d0cdc5ac700c493747f5b35fe/python-sdk/nuscenes/utils/splits.py#L189
#     - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
#     - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
#     - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
#         tracker if required
DATASET_NAME_TO_VERSION = {
    "nusc_train": "v1.0-trainval",
    "nusc_val": "v1.0-trainval",
    "nusc_val-subsample-8": "v1.0-trainval",
    "nusc_trainval": "v1.0-trainval",
    "nusc_test": "v1.0-test",
    "nusc_mini_train": "v1.0-mini",
    "nusc_mini_val": "v1.0-mini",
}

CAMERA_NAMES = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')

ATTRIBUTE_IDS = {
    'vehicle.moving': 0,
    'vehicle.parked': 1,
    'vehicle.stopped': 2,
    'pedestrian.moving': 0,
    'pedestrian.standing': 1,
    'pedestrian.sitting_lying_down': 2,
    'cycle.with_rider': 0,
    'cycle.without_rider': 1,
}

CATEGORY_IDS = OrderedDict({
    'barrier': 0,
    'bicycle': 1,
    'bus': 2,
    'car': 3,
    'construction_vehicle': 4,
    'motorcycle': 5,
    'pedestrian': 6,
    'traffic_cone': 7,
    'trailer': 8,
    'truck': 9,
})

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=10)]
COLORMAP = OrderedDict({
    'barrier': COLORS[8],  # yellow
    'bicycle': COLORS[0],  # blue
    'bus': COLORS[6],  # pink
    'car': COLORS[2],  # green
    'construction_vehicle': COLORS[7],  # gray
    'motorcycle': COLORS[4],  # purple
    'pedestrian': COLORS[1],  # orange
    'traffic_cone': COLORS[3],  # red
    'trailer': COLORS[9],  # skyblue
    'truck': COLORS[5],  # brown
})

MAX_NUM_ATTRIBUTES = 3


def _compute_iou(box1, box2):
    """
    Parameters
    ----------
    box1, box2:
        (x1, y1, x2, y2)
    """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    if xx1 >= xx2 or yy1 >= yy2:
        return 0.
    inter = (xx2 - xx1) * (yy2 - yy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


class NuscenesDataset(Dataset):
    def __init__(self, name, data_root, datum_names=CAMERA_NAMES, min_num_lidar_points=3, min_box_visibility=0.2, **unused):
        self.data_root = data_root
        assert name in DATASET_NAME_TO_VERSION
        version = DATASET_NAME_TO_VERSION[name]
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

        self.datum_names = datum_names
        self.min_num_lidar_points = min_num_lidar_points
        self.min_box_visibility = min_box_visibility

        self.dataset_item_info = self._build_dataset_item_info(name)

        # Index instance tokens to their IDs
        self._instance_token_to_id = self._index_instance_tokens()

        # Construct the mapping from datum_token (image id) to index
        print("Generating the mapping from image id to idx...")
        self.datumtoken2idx = {}
        for idx, (datum_token, _, _, _, _) in enumerate(self.dataset_item_info):
            self.datumtoken2idx[datum_token] = idx
        print("Done.")

    def _build_dataset_item_info(self, name):
        scenes_in_split = self._get_split_scenes(name)

        dataset_items = []
        for _, scene_token in tqdm(scenes_in_split):
            scene = self.nusc.get('scene', scene_token)
            sample_token = scene['first_sample_token']
            for sample_idx in range(scene['nbr_samples']):
                if name.endswith('subsample-8') and sample_idx % 8 > 0:
                    # Sample-level subsampling.
                    continue

                sample = self.nusc.get('sample', sample_token)
                for datum_name, datum_token in sample['data'].items():
                    if datum_name not in self.datum_names:
                        continue
                    dataset_items.append((datum_token, sample_token, scene['name'], sample_idx, datum_name))
                sample_token = sample['next']
        return dataset_items

    def _get_split_scenes(self, name):
        scenes_in_splits = create_splits_scenes()
        if name == "nusc_trainval":
            scenes = scenes_in_splits["train"] + scenes_in_splits["val"]
        elif name == "nusc_val-subsample-8":
            scenes = scenes_in_splits["val"]
        else:
            assert name.startswith('nusc_'), f"Invalid dataset name: {name}"
            split = name[5:]
            assert split in scenes_in_splits, f"Invalid dataset: {split}"
            scenes = scenes_in_splits[split]

        # Mapping from scene name to token.
        name_to_token = {scene['name']: scene['token'] for scene in self.nusc.scene}
        return [(name, name_to_token[name]) for name in scenes]

    def __len__(self):
        return len(self.dataset_item_info)

    def _build_id(self, scene_name, sample_idx, datum_name):
        sample_id = f"{scene_name}_{sample_idx:03d}"
        image_id = f"{sample_id}_{datum_name}"
        return image_id, sample_id

    def _index_instance_tokens(self):
        """Index instance tokens for uniquely identifying instances across samples"""
        instance_token_to_id = {}
        for record in self.nusc.sample_annotation:
            instance_token = record['instance_token']
            if instance_token not in instance_token_to_id:
                next_instance_id = len(instance_token_to_id)
                instance_token_to_id[instance_token] = next_instance_id
        return instance_token_to_id

    def get_instance_annotations(self, annotation_list, K, image_shape, pose_WS):
        annotations = []
        for _ann in annotation_list:
            ann = self.nusc.get('sample_annotation', _ann.token)
            if ann['num_lidar_pts'] + ann['num_radar_pts'] < self.min_num_lidar_points:
                continue
            annotation = OrderedDict()

            # --------
            # Category
            # --------
            category = category_to_detection_name(ann['category_name'])
            if category is None:
                continue
            annotation['category_id'] = CATEGORY_IDS[category]

            # ------
            # 3D box
            # ------
            # NOTE: ann['rotation'], ann['translation'] is in global frame.
            pose_SO = Pose(wxyz=_ann.orientation, tvec=_ann.center)  # pose in sensor frame
            # DEBUG:
            # pose_WO_1 = Pose(np.array(ann['rotation']), np.array(ann['translation']))
            # pose_WO_2 = pose_WS * pose_SO
            # assert np.allclose(pose_WO_1.matrix, pose_WO_2.matrix)
            bbox3d = GenericBoxes3D(_ann.orientation, _ann.center, _ann.wlh)
            annotation['bbox3d'] = bbox3d.vectorize().tolist()[0]

            # --------------------------------------
            # 2D box -- project 8 corners of 3D bbox
            # --------------------------------------
            corners = project_points3d(bbox3d.corners.cpu().numpy().squeeze(0), K)
            l, t = corners[:, 0].min(), corners[:, 1].min()
            r, b = corners[:, 0].max(), corners[:, 1].max()

            x1 = max(0, l)
            y1 = max(0, t)
            x2 = min(image_shape[1], r)
            y2 = min(image_shape[0], b)

            iou = _compute_iou([l, t, r, b], [x1, y1, x2, y2])
            if iou < self.min_box_visibility:
                continue

            annotation['bbox'] = [x1, y1, x2, y2]
            annotation['bbox_mode'] = BoxMode.XYXY_ABS

            # --------
            # Track ID
            # --------
            annotation['track_id'] = self._instance_token_to_id[ann['instance_token']]

            # ---------
            # Attribute
            # ---------
            attr_tokens = ann['attribute_tokens']
            assert len(attr_tokens) < 2  # NOTE: Allow only single attrubute.
            attribute_id = MAX_NUM_ATTRIBUTES  # By default, MAX_NUM_ATTRIBUTES -- this is to be ignored in loss compute.
            if attr_tokens:
                attribute = self.nusc.get('attribute', attr_tokens[0])['name']
                attribute_id = ATTRIBUTE_IDS[attribute]
            annotation['attribute_id'] = attribute_id

            # -----
            # Speed
            # -----
            vel_global = self.nusc.box_velocity(ann['token'])
            speed = np.linalg.norm(vel_global)  # NOTE: This can be NaN.
            # DEBUG:
            # speed * Quaternion(ann['rotation']).rotation_matrix.T[0] ~= vel_global
            annotation['speed'] = speed

            annotations.append(annotation)

        return annotations

    def _get_ego_velocity(self, current, max_time_diff=1.5):
        """Velocity of ego-vehicle in m/s.
        """
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.nusc.get('sample_data', current['prev'])
        else:
            first = current

        if has_next:
            last = self.nusc.get('sample_data', current['next'])
        else:
            last = current

        pos_first = self.nusc.get('ego_pose', first['ego_pose_token'])['translation']
        pos_last = self.nusc.get('ego_pose', last['ego_pose_token'])['translation']
        pos_diff = np.float32(pos_last) - np.float32(pos_first)

        time_last = 1e-6 * last['timestamp']
        time_first = 1e-6 * first['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    def __getitem__(self, idx):
        datum_token, sample_token, scene_name, sample_idx, datum_name = self.dataset_item_info[idx]
        datum = self.nusc.get('sample_data', datum_token)
        assert datum['is_key_frame']

        filename, _annotations, K = self.nusc.get_sample_data(datum_token)
        image_id, sample_id = self._build_id(scene_name, sample_idx, datum_name)
        height, width = datum['height'], datum['width']
        d2_dict = OrderedDict(
            file_name=filename,
            height=height,
            width=width,
            image_id=image_id,
            sample_id=sample_id,
            sample_token=sample_token
        )

        # Intrinsics
        d2_dict['intrinsics'] = list(K.flatten())

        # Get pose of the sensor (S) from vehicle (V) frame
        _pose_VS = self.nusc.get('calibrated_sensor', datum['calibrated_sensor_token'])
        pose_VS = Pose(wxyz=np.float64(_pose_VS['rotation']), tvec=np.float64(_pose_VS['translation']))

        # Get ego-pose of the vehicle (V) from global/world (W) frame
        _pose_WV = self.nusc.get('ego_pose', datum['ego_pose_token'])
        pose_WV = Pose(wxyz=np.float64(_pose_WV['rotation']), tvec=np.float64(_pose_WV['translation']))
        pose_WS = pose_WV * pose_VS

        d2_dict['pose'] = {'wxyz': list(pose_WS.quat.elements), 'tvec': list(pose_WS.tvec)}
        d2_dict['extrinsics'] = {'wxyz': list(pose_VS.quat.elements), 'tvec': list(pose_VS.tvec)}

        d2_dict['ego_speed'] = np.linalg.norm(self._get_ego_velocity(datum))

        d2_dict['annotations'] = self.get_instance_annotations(_annotations, K, (height, width), pose_WS)

        return d2_dict

    def getitem_by_datumtoken(self, datum_token):
        # idx = self.datumtoken2idx[datum_token]
        # ret = self.__getitem__(idx)

        datum = self.nusc.get('sample_data', datum_token)
        sample_token = datum['sample_token']
        filename, _annotations, K = self.nusc.get_sample_data(datum_token)
        height, width = datum['height'], datum['width']
        d2_dict = OrderedDict(
            file_name=filename,
            height=height,
            width=width,
            image_id=0,
            sample_id=0,
            sample_token=sample_token
        )
        # Intrinsics
        d2_dict['intrinsics'] = list(K.flatten())
        # Get pose of the sensor (S) from vehicle (V) frame
        _pose_VS = self.nusc.get('calibrated_sensor', datum['calibrated_sensor_token'])
        pose_VS = Pose(wxyz=np.float64(_pose_VS['rotation']), tvec=np.float64(_pose_VS['translation'])) 
        # Get ego-pose of the vehicle (V) from global/world (W) frame
        _pose_WV = self.nusc.get('ego_pose', datum['ego_pose_token'])
        pose_WV = Pose(wxyz=np.float64(_pose_WV['rotation']), tvec=np.float64(_pose_WV['translation']))
        pose_WS = pose_WV * pose_VS

        d2_dict['pose'] = {'wxyz': list(pose_WS.quat.elements), 'tvec': list(pose_WS.tvec)}
        d2_dict['extrinsics'] = {'wxyz': list(pose_VS.quat.elements), 'tvec': list(pose_VS.tvec)}

        d2_dict['ego_speed'] = np.linalg.norm(self._get_ego_velocity(datum))

        d2_dict['annotations'] = self.get_instance_annotations(_annotations, K, (height, width), pose_WS)
        return d2_dict