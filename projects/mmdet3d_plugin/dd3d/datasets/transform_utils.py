# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from detectron2:
#   https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/detection_utils.py
import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances

from projects.mmdet3d_plugin.dd3d.structures.boxes3d import Boxes3D

__all__ = ["transform_instance_annotations", "annotations_to_instances"]


def transform_instance_annotations(
    annotation,
    transforms,
    image_size,
):
    """Adapted from:
        https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/detection_utils.py#L254

    The changes from original:
        - The presence of 2D bounding box (i.e. "bbox" field) is assumed by default in d2; here it's optional.
        - Add optional 3D bounding box support.
        - If the instance mask annotation is in RLE, then it's decoded into polygons, not bitmask, to save memory.

    ===============================================================================================================

    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # (dennis.park) Here 2D bounding box is optional.
    if "bbox" in annotation:
        assert "bbox_mode" in annotation, "'bbox' is present, but 'bbox_mode' is not."
        # bbox is 1d (per-instance bounding box)
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        bbox = transforms.apply_box(np.array([bbox]))[0]
        # clip transformed bbox to image size
        bbox = bbox.clip(min=0)
        bbox = np.minimum(bbox, list(image_size + image_size)[::-1])
        annotation["bbox"] = bbox
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

    # Vertical flipping is not implemented (`flip_transform.py`). TODO: implement if needed.
    if "bbox3d" in annotation:
        bbox3d = np.array(annotation["bbox3d"])
        annotation['bbox3d'] = transforms.apply_box3d(bbox3d)

    return annotation


def _create_empty_instances(image_size):
    target = Instances(image_size)

    target.gt_boxes = Boxes([])
    target.gt_classes = torch.tensor([], dtype=torch.int64)
    target.gt_boxes3d = Boxes3D.from_vectors([], torch.eye(3, dtype=torch.float32))

    return target


def annotations_to_instances(
    annos,
    image_size,
    intrinsics=None,
):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    if len(annos) == 0:
        return _create_empty_instances(image_size)

    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "bbox3d" in annos[0]:
        assert intrinsics is not None
        target.gt_boxes3d = Boxes3D.from_vectors([anno['bbox3d'] for anno in annos], intrinsics)
        if len(target.gt_boxes3d) != target.gt_boxes.tensor.shape[0]:
            raise ValueError(
                f"The sizes of `gt_boxes3d` and `gt_boxes` do not match: a={len(target.gt_boxes3d)}, b={target.gt_boxes.tensor.shape[0]}."
            )

    # NOTE: add nuscenes attributes here
    # NOTE: instances will be filtered later
    # NuScenes attributes
    if len(annos) and "attribute_id" in annos[0]:    
        attributes = [obj["attribute_id"] for obj in annos] 
        target.gt_attributes = torch.tensor(attributes, dtype=torch.int64)

    # Speed (magnitude of velocity)
    if len(annos) and "speed" in annos[0]:
        speeds = [obj["speed"] for obj in annos]
        target.gt_speeds = torch.tensor(speeds, dtype=torch.float32)

    assert len(boxes) == len(classes) == len(attributes) == len(speeds), \
        'the numbers of annotations should be the same'
    return target
