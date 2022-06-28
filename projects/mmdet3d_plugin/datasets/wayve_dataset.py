import numpy as np
from os import path as osp

import pickle
from mmdet3d.core import show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes


@DATASETS.register_module()
class WayveDataset(Custom3DDataset):
    CLASSES = (
        'car', 'bus', 'van', 'truck', 'bicycle', 'motorcycle', 'scooter',
        'cyclist', 'motorcyclist', 'scooterist', 'pedestrian', 'traffic_light',
        'unknown',
    )
    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        pass
        #  with open(ann_file, 'rb') as f:
            #  return pickle.load(f)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=info['pts_filename'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp_us'] / 1e6,
        )
        # For wayve dataset, the cuboids are labelled in the vehicle frame, so we don't need to use the lidar sensor's
        # pose
        # However, to make it work with the rest of the code, let's still call it 'lidar2cam'
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cameras'].items():
                image_paths.append(cam_info['path'])
                # obtain lidar to image transformation matrix
                cam2ego = np.array(cam_info['pose'])
                ego2cam = np.linalg.inv(cam2ego)
                intrinsic = np.array(cam_info['intrinsics'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ ego2cam)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(ego2cam)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        # filter out bbox containing no points
        mask = np.array(info['num_pts']) > 0
        pos = np.array(info['pos'])[mask]
        size = np.array(info['size'])[mask]
        yaw_deg = np.array(info['yaw_deg'])[mask]
        # Convert yaw to lidar box format - for us it's
        #
        #                             up z    x front (yaw_deg=0)
        #                               ^   ^
        #                               |  /
        #                               | /
        #      (yaw=-pi) left y <------ 0 -------- (yaw_deg=-90)
        yaw = yaw_deg * np.pi / 180
        yaw = -yaw - np.pi / 2
        gt_bboxes_3d = np.concatenate([pos, size, yaw[:, None]], axis=1)
        gt_names_3d = np.array(info['labels'])[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the wayve box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d
        )
        return anns_results

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir, file_name, show)
