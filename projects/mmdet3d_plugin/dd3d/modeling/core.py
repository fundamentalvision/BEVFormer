# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch
from torch import nn

#from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess as resize_instances
from detectron2.structures import Instances
from detectron2.layers import ShapeSpec
from mmcv.runner import force_fp32

from .fcos2d import FCOS2DHead, FCOS2DInference, FCOS2DLoss
from .fcos3d import FCOS3DHead, FCOS3DInference, FCOS3DLoss
#from tridet.modeling.dd3d.postprocessing import nuscenes_sample_aggregate
from .prepare_targets import DD3DTargetPreparer
#from tridet.modeling.feature_extractor import build_feature_extractor
from projects.mmdet3d_plugin.dd3d.structures.image_list import ImageList
from projects.mmdet3d_plugin.dd3d.utils.tensor2d import compute_features_locations as compute_locations_per_level


#@META_ARCH_REGISTRY.register()
class DD3D(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 strides,
                 fcos2d_cfg=dict(),
                 fcos2d_loss_cfg=dict(),
                 fcos3d_cfg=dict(),
                 fcos3d_loss_cfg=dict(),
                 target_assign_cfg=dict(),
                 box3d_on=True,
                 feature_locations_offset="none"):
        super().__init__()
        # NOTE: do not need backbone
        # self.backbone = build_feature_extractor(cfg)
        # backbone_output_shape = self.backbone.output_shape()
        # self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        
        self.backbone_output_shape = [ShapeSpec(channels=in_channels, stride=s) for s in strides]

        self.feature_locations_offset = feature_locations_offset

        self.fcos2d_head = FCOS2DHead(num_classes=num_classes, input_shape=self.backbone_output_shape,
                                     **fcos2d_cfg)
        self.fcos2d_loss = FCOS2DLoss(num_classes=num_classes, **fcos2d_loss_cfg)
        # NOTE: inference later
        # self.fcos2d_inference = FCOS2DInference(cfg)

        if box3d_on:
            self.fcos3d_head = FCOS3DHead(num_classes=num_classes, input_shape=self.backbone_output_shape,
                                          **fcos3d_cfg)
            self.fcos3d_loss = FCOS3DLoss(num_classes=num_classes, **fcos3d_loss_cfg)
            # NOTE: inference later
            # self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        self.prepare_targets = DD3DTargetPreparer(num_classes=num_classes, 
                                                  input_shape=self.backbone_output_shape,
                                                  box3d_on=box3d_on,
                                                  **target_assign_cfg)

        # NOTE: inference later
        # self.postprocess_in_inference = cfg.DD3D.INFERENCE.DO_POSTPROCESS

        # self.do_nms = cfg.DD3D.INFERENCE.DO_NMS
        # self.do_bev_nms = cfg.DD3D.INFERENCE.DO_BEV_NMS
        # self.bev_nms_iou_thresh = cfg.DD3D.INFERENCE.BEV_NMS_IOU_THRESH

        # nuScenes inference aggregates detections over all 6 cameras.
        # self.nusc_sample_aggregate_in_inference = cfg.DD3D.INFERENCE.NUSC_SAMPLE_AGGREGATE
        self.num_classes = num_classes

        # NOTE: do not need normalize
        # self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        # self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    # NOTE:
    # @property
    # def device(self):
    #     return self.pixel_mean.device

    # def preprocess_image(self, x):
    #     return (x - self.pixel_mean) / self.pixel_std

    @force_fp32(apply_to=('features'))
    def forward(self, features, batched_inputs):
        # NOTE:
        # images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [self.preprocess_image(x) for x in images]

        # NOTE: directly use inv_intrinsics
        # if 'intrinsics' in batched_inputs[0]:
        #     intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        # else:
        #     intrinsics = None
        # images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)
        if 'inv_intrinsics' in batched_inputs[0]:
            inv_intrinsics = [x['inv_intrinsics'].to(features[0].device) for x in batched_inputs]
            inv_intrinsics = torch.stack(inv_intrinsics, dim=0)
        else:
            inv_intrinsics = None

        # NOTE:
        # gt_dense_depth = None
        # if 'depth' in batched_inputs[0]:
        #     gt_dense_depth = [x["depth"].to(self.device) for x in batched_inputs]
        #     gt_dense_depth = ImageList.from_tensors(
        #         gt_dense_depth, self.backbone.size_divisibility, intrinsics=intrinsics
        #     )

        # NOTE: directly input feature
        # features = self.backbone(images.tensor)
        # features = [features[f] for f in self.in_features]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(features[0].device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth = self.fcos3d_head(features)
        # NOTE: directly use inv_intrinsics
        # inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        if self.training:
            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            # NOTE: 
            # if gt_dense_depth is not None:
            #    training_targets.update({"dense_depth": gt_dense_depth})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)
            return losses
        else:
            # TODO: do not support inference now
            raise NotImplementedError
            
            pred_instances, fcos2d_info = self.fcos2d_inference(
                logits, box2d_reg, centerness, locations, images.image_sizes
            )
            if not self.only_box2d:
                # This adds 'pred_boxes3d' and 'scores_3d' to Instances in 'pred_instances' in place.
                self.fcos3d_inference(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, inv_intrinsics, pred_instances,
                    fcos2d_info
                )

                # 3D score == 2D score x confidence.
                score_key = "scores_3d"
            else:
                score_key = "scores"

            # Transpose to "image-first", i.e. (B, L)
            pred_instances = list(zip(*pred_instances))
            pred_instances = [Instances.cat(instances) for instances in pred_instances]

            # 2D NMS and pick top-K.
            if self.do_nms:
                pred_instances = self.fcos2d_inference.nms_and_top_k(pred_instances, score_key)

            if not self.only_box2d and self.do_bev_nms:
                # Bird-eye-view NMS.
                dummy_group_idxs = {i: [i] for i, _ in enumerate(pred_instances)}
                if 'pose' in batched_inputs[0]:
                    poses = [x['pose'] for x in batched_inputs]
                else:
                    poses = [x['extrinsics'] for x in batched_inputs]
                pred_instances = nuscenes_sample_aggregate(
                    pred_instances,
                    dummy_group_idxs,
                    self.num_classes,
                    poses,
                    iou_threshold=self.bev_nms_iou_thresh,
                    include_boxes3d_global=False
                )

            if self.postprocess_in_inference:
                processed_results = []
                for results_per_image, input_per_image, image_size in \
                        zip(pred_instances, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = resize_instances(results_per_image, height, width)
                    processed_results.append({"instances": r})
            else:
                processed_results = [{"instances": x} for x in pred_instances]

            return processed_results

    def compute_locations(self, features):
        locations = []
        in_strides = [x.stride for x in self.backbone_output_shape]
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations_per_level(
                h, w, in_strides[level], feature.dtype, feature.device, offset=self.feature_locations_offset
            )
            locations.append(locations_per_level)
        return locations

    def forward_train(self, features, batched_inputs):
        self.train()
        return self.forward(features, batched_inputs)