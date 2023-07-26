import copy
import numpy as np
import torch
from mmcv.parallel.data_container import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from projects.mmdet3d_plugin.dd3d.datasets.transform_utils import annotations_to_instances
from projects.mmdet3d_plugin.dd3d.structures.pose import Pose
from projects.mmdet3d_plugin.dd3d.utils.tasks import TaskManager


@PIPELINES.register_module()
class DD3DMapper:
    def __init__(self,
                 is_train: bool = True,
                 tasks=dict(box2d_on=True, box3d_on=True),
                 ):
        self.is_train = is_train
        self.task_manager = TaskManager(**tasks)

    def __call__(self, results):
        if results['mono_input_dict'] is None:
            return results
        mono_input_dict = []
        for dataset_dict in results['mono_input_dict']:
            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
            image_shape = results['img'].data.shape[-2:]
            intrinsics = None
            if "intrinsics" in dataset_dict:
                intrinsics = dataset_dict['intrinsics']
                if not torch.is_tensor(intrinsics):
                    intrinsics = np.reshape(
                        intrinsics,
                        (3, 3),
                    ).astype(np.float32)
                    intrinsics = torch.as_tensor(intrinsics)
                    # NOTE: intrinsics = transforms.apply_intrinsics(intrinsics)
                    dataset_dict["intrinsics"] = intrinsics
                dataset_dict["inv_intrinsics"] = torch.linalg.inv(dataset_dict['intrinsics'])

            if "pose" in dataset_dict:
                pose = Pose(wxyz=np.float32(dataset_dict["pose"]["wxyz"]),
                            tvec=np.float32(dataset_dict["pose"]["tvec"]))
                dataset_dict["pose"] = pose
                # NOTE: no transforms affect global pose.

            if "extrinsics" in dataset_dict:
                extrinsics = Pose(
                    wxyz=np.float32(dataset_dict["extrinsics"]["wxyz"]),
                    tvec=np.float32(dataset_dict["extrinsics"]["tvec"])
                )
                dataset_dict["extrinsics"] = extrinsics

            if not self.task_manager.has_detection_task:
                dataset_dict.pop("annotations", None)

            if "annotations" in dataset_dict:
                for anno in dataset_dict["annotations"]:
                    if not self.task_manager.has_detection_task:
                        anno.pop("bbox", None)
                        anno.pop("bbox_mode", None)
                    if not self.task_manager.box3d_on:
                        anno.pop("bbox3d", None)
                annos = [anno for anno in dataset_dict["annotations"] if anno.get("iscrowd", 0) == 0]
                if annos and 'bbox3d' in annos[0]:
                    # Remove boxes with negative z-value for center.
                    annos = [anno for anno in annos if anno['bbox3d'][6] > 0]

                instances = annotations_to_instances(
                    annos,
                    image_shape,  # TODO: the effect of the shape?
                    intrinsics=intrinsics.numpy(),
                )

                if self.is_train:
                    # instances = d2_utils.filter_empty_instances(instances)
                    m = instances.gt_boxes.nonempty(threshold=1e-5)
                    instances = instances[m]
                    annos = [anno for tmp_m, anno in zip(m, annos) if tmp_m]
                dataset_dict["instances"] = instances

            dataset_dict['annotations'] = annos

            mono_input_dict.append(dataset_dict)

        # TODO: drop batch that has no annotations?
        box_num = 0
        for dataset_dict in mono_input_dict:
            box_num += dataset_dict["instances"].gt_boxes.tensor.shape[0]
        if box_num == 0:
            return None

        mono_input_dict = DC(mono_input_dict, cpu_only=True)
        results['mono_input_dict'] = mono_input_dict
        return results
