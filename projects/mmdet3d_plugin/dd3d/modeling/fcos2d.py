# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from AdelaiDet:
#   https://github.com/aim-uofa/AdelaiDet
import torch
from fvcore.nn import sigmoid_focal_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, batched_nms, cat, get_norm
from detectron2.structures import Boxes, Instances
from detectron2.utils.comm import get_world_size
from mmcv.runner import force_fp32

from projects.mmdet3d_plugin.dd3d.layers.iou_loss import IOULoss
from projects.mmdet3d_plugin.dd3d.layers.normalization import ModuleListDial, Scale
from projects.mmdet3d_plugin.dd3d.utils.comm import reduce_sum

INF = 100000000


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class FCOS2DHead(nn.Module):
    def __init__(self, 
                 num_classes, 
                 input_shape,
                 num_cls_convs=4,
                 num_box_convs=4,
                 norm='BN',
                 use_deformable=False,
                 use_scale=True,
                 box2d_scale_init_factor=1.0,
                 version='v2'):
        super().__init__()

        self.num_classes = num_classes
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)

        self.use_scale = use_scale
        self.box2d_scale_init_factor = box2d_scale_init_factor

        self._version = version

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if use_deformable:
            raise ValueError("Not supported yet.")

        head_configs = {'cls': num_cls_convs, 'box2d': num_box_convs}

        for head_name, num_convs in head_configs.items():
            tower = []
            if self._version == "v1":
                for _ in range(num_convs):
                    conv_func = nn.Conv2d
                    tower.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
                    if norm == "GN":
                        raise NotImplementedError()
                    elif norm == "NaiveGN":
                        raise NotImplementedError()
                    elif norm == "BN":
                        tower.append(ModuleListDial([nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)]))
                    elif norm == "SyncBN":
                        raise NotImplementedError()
                    tower.append(nn.ReLU())
            elif self._version == "v2":
                for _ in range(num_convs):
                    if norm in ("BN", "FrozenBN", "SyncBN", "GN"):
                        # NOTE: need to add norm here!
                        # Each FPN level has its own batchnorm layer.
                        # NOTE: do not use dd3d train.py!
                        # "BN" is converted to "SyncBN" in distributed training (see train.py)
                        norm_layer = ModuleListDial([get_norm(norm, in_channels) for _ in range(self.num_levels)])
                    else:
                        norm_layer = get_norm(norm, in_channels)
                    tower.append(
                        Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=norm_layer is None,
                            norm=norm_layer,
                            activation=F.relu
                        )
                    )
            else:
                raise ValueError(f"Invalid FCOS2D version: {self._version}")
            self.add_module(f'{head_name}_tower', nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.box2d_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if self.use_scale:
            if self._version == "v1":
                self.scales_reg = nn.ModuleList([
                    Scale(init_value=stride * self.box2d_scale_init_factor) for stride in self.in_strides
                ])
            else:
                self.scales_box2d_reg = nn.ModuleList([
                    Scale(init_value=stride * self.box2d_scale_init_factor) for stride in self.in_strides
                ])

        self.init_weights()

    def init_weights(self):

        for tower in [self.cls_tower, self.box2d_tower]:
            for l in tower.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        predictors = [self.cls_logits, self.box2d_reg, self.centerness]

        for modules in predictors:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        box2d_reg = []
        centerness = []

        extra_output = {"cls_tower_out": []}

        for l, feature in enumerate(x):
            cls_tower_out = self.cls_tower(feature)
            bbox_tower_out = self.box2d_tower(feature)

            # 2D box
            logits.append(self.cls_logits(cls_tower_out))
            centerness.append(self.centerness(bbox_tower_out))
            box_reg = self.box2d_reg(bbox_tower_out)
            if self.use_scale:
                # TODO: to optimize the runtime, apply this scaling in inference (and loss compute) only on FG pixels?
                if self._version == "v1":
                    box_reg = self.scales_reg[l](box_reg)
                else:
                    box_reg = self.scales_box2d_reg[l](box_reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            box2d_reg.append(F.relu(box_reg))

            extra_output['cls_tower_out'].append(cls_tower_out)

        return logits, box2d_reg, centerness, extra_output


class FCOS2DLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 loc_loss_type='giou',
                 ):
        super().__init__()
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        self.box2d_reg_loss_fn = IOULoss(loc_loss_type)

        self.num_classes = num_classes

    @force_fp32(apply_to=('logits', 'box2d_reg', 'centerness'))
    def forward(self, logits, box2d_reg, centerness, targets):
        labels = targets['labels']
        box2d_reg_targets = targets['box2d_reg_targets']
        pos_inds = targets["pos_inds"]

        if len(labels) != box2d_reg_targets.shape[0]:
            raise ValueError(
                f"The size of 'labels' and 'box2d_reg_targets' does not match: a={len(labels)}, b={box2d_reg_targets.shape[0]}"
            )

        # Flatten predictions
        logits = cat([x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits])
        box2d_reg_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in box2d_reg])
        centerness_pred = cat([x.permute(0, 2, 3, 1).reshape(-1) for x in centerness])

        # -------------------
        # Classification loss
        # -------------------
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        cls_target = torch.zeros_like(logits)
        cls_target[pos_inds, labels[pos_inds]] = 1

        loss_cls = sigmoid_focal_loss(
            logits,
            cls_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        # NOTE: The rest of losses only consider foreground pixels.
        box2d_reg_pred = box2d_reg_pred[pos_inds]
        box2d_reg_targets = box2d_reg_targets[pos_inds]

        centerness_pred = centerness_pred[pos_inds]

        # Compute centerness targets here using 2D regression targets of foreground pixels.
        centerness_targets = compute_ctrness_targets(box2d_reg_targets)

        # Denominator for all foreground losses.
        ctrness_targets_sum = centerness_targets.sum()
        loss_denom = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

        # NOTE: change the return after reduce_sum
        if pos_inds.numel() == 0:
            losses = {
                "loss_cls": loss_cls,
                "loss_box2d_reg": box2d_reg_pred.sum() * 0.,
                "loss_centerness": centerness_pred.sum() * 0.,
            }
            return losses, {}

        # ----------------------
        # 2D box regression loss
        # ----------------------
        loss_box2d_reg = self.box2d_reg_loss_fn(box2d_reg_pred, box2d_reg_targets, centerness_targets) / loss_denom

        # ---------------
        # Centerness loss
        # ---------------
        loss_centerness = F.binary_cross_entropy_with_logits(
            centerness_pred, centerness_targets, reduction="sum"
        ) / num_pos_avg

        loss_dict = {"loss_cls": loss_cls, "loss_box2d_reg": loss_box2d_reg, "loss_centerness": loss_centerness}
        extra_info = {"loss_denom": loss_denom, "centerness_targets": centerness_targets}

        return loss_dict, extra_info


class FCOS2DInference():
    def __init__(self, cfg):
        self.thresh_with_ctr = cfg.DD3D.FCOS2D.INFERENCE.THRESH_WITH_CTR
        self.pre_nms_thresh = cfg.DD3D.FCOS2D.INFERENCE.PRE_NMS_THRESH
        self.pre_nms_topk = cfg.DD3D.FCOS2D.INFERENCE.PRE_NMS_TOPK
        self.post_nms_topk = cfg.DD3D.FCOS2D.INFERENCE.POST_NMS_TOPK
        self.nms_thresh = cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH
        self.num_classes = cfg.DD3D.NUM_CLASSES

    def __call__(self, logits, box2d_reg, centerness, locations, image_sizes):

        pred_instances = []  # List[List[Instances]], shape = (L, B)
        extra_info = []
        for lvl, (logits_lvl, box2d_reg_lvl, centerness_lvl, locations_lvl) in \
            enumerate(zip(logits, box2d_reg, centerness, locations)):

            instances_per_lvl, extra_info_per_lvl = self.forward_for_single_feature_map(
                logits_lvl, box2d_reg_lvl, centerness_lvl, locations_lvl, image_sizes
            )  # List of Instances; one for each image.

            for instances_per_im in instances_per_lvl:
                instances_per_im.fpn_levels = locations_lvl.new_ones(len(instances_per_im), dtype=torch.long) * lvl

            pred_instances.append(instances_per_lvl)
            extra_info.append(extra_info_per_lvl)

        return pred_instances, extra_info

    def forward_for_single_feature_map(self, logits, box2d_reg, centerness, locations, image_sizes):
        N, C, _, __ = logits.shape

        # put in the same format as locations
        scores = logits.permute(0, 2, 3, 1).reshape(N, -1, C).sigmoid()
        box2d_reg = box2d_reg.permute(0, 2, 3, 1).reshape(N, -1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(N, -1).sigmoid()

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            scores = scores * centerness[:, :, None]

        candidate_mask = scores > self.pre_nms_thresh

        pre_nms_topk = candidate_mask.reshape(N, -1).sum(1)
        pre_nms_topk = pre_nms_topk.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            scores = scores * centerness[:, :, None]

        results = []
        all_fg_inds_per_im, all_topk_indices, all_class_inds_per_im = [], [], []
        for i in range(N):
            scores_per_im = scores[i]
            candidate_mask_per_im = candidate_mask[i]
            scores_per_im = scores_per_im[candidate_mask_per_im]

            candidate_inds_per_im = candidate_mask_per_im.nonzero(as_tuple=False)
            fg_inds_per_im = candidate_inds_per_im[:, 0]
            class_inds_per_im = candidate_inds_per_im[:, 1]

            # Cache info here.
            all_fg_inds_per_im.append(fg_inds_per_im)
            all_class_inds_per_im.append(class_inds_per_im)

            box2d_reg_per_im = box2d_reg[i][fg_inds_per_im]
            locations_per_im = locations[fg_inds_per_im]

            pre_nms_topk_per_im = pre_nms_topk[i]

            if candidate_mask_per_im.sum().item() > pre_nms_topk_per_im.item():
                scores_per_im, topk_indices = \
                    scores_per_im.topk(pre_nms_topk_per_im, sorted=False)

                class_inds_per_im = class_inds_per_im[topk_indices]
                box2d_reg_per_im = box2d_reg_per_im[topk_indices]
                locations_per_im = locations_per_im[topk_indices]
            else:
                topk_indices = None

            all_topk_indices.append(topk_indices)

            detections = torch.stack([
                locations_per_im[:, 0] - box2d_reg_per_im[:, 0],
                locations_per_im[:, 1] - box2d_reg_per_im[:, 1],
                locations_per_im[:, 0] + box2d_reg_per_im[:, 2],
                locations_per_im[:, 1] + box2d_reg_per_im[:, 3],
            ],
                                     dim=1)

            instances = Instances(image_sizes[i])
            instances.pred_boxes = Boxes(detections)
            instances.scores = torch.sqrt(scores_per_im)
            instances.pred_classes = class_inds_per_im
            instances.locations = locations_per_im

            results.append(instances)

        extra_info = {
            "fg_inds_per_im": all_fg_inds_per_im,
            "class_inds_per_im": all_class_inds_per_im,
            "topk_indices": all_topk_indices
        }
        return results, extra_info

    def nms_and_top_k(self, instances_per_im, score_key_for_nms="scores"):
        results = []
        for instances in instances_per_im:
            if self.nms_thresh > 0:
                # Multiclass NMS.
                keep = batched_nms(
                    instances.pred_boxes.tensor, instances.get(score_key_for_nms), instances.pred_classes,
                    self.nms_thresh
                )
                instances = instances[keep]
            num_detections = len(instances)

            # Limit to max_per_image detections **over all classes**
            if num_detections > self.post_nms_topk > 0:
                scores = instances.scores
                # image_thresh, _ = torch.kthvalue(scores.cpu(), num_detections - self.post_nms_topk + 1)
                image_thresh, _ = torch.kthvalue(scores, num_detections - self.post_nms_topk + 1)
                keep = scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                instances = instances[keep]
            results.append(instances)
        return results
