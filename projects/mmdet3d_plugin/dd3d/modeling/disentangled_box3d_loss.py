# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging

import torch
import torch.nn as nn

from projects.mmdet3d_plugin.dd3d.layers.smooth_l1_loss import smooth_l1_loss

LOG = logging.getLogger(__name__)


class DisentangledBox3DLoss(nn.Module):
    def __init__(self, smooth_l1_loss_beta, max_loss_per_group):
        super().__init__()
        self.smooth_l1_loss_beta = smooth_l1_loss_beta
        self.max_loss_per_group = max_loss_per_group

    def forward(self, box3d_pred, box3d_targets, locations, weights=None):

        box3d_pred = box3d_pred.to(torch.float32)
        box3d_targets = box3d_targets.to(torch.float32)

        target_corners = box3d_targets.corners

        disentangled_losses = {}
        for component_key in ["quat", "proj_ctr", "depth", "size"]:
            disentangled_boxes = box3d_targets.clone()
            setattr(disentangled_boxes, component_key, getattr(box3d_pred, component_key))
            pred_corners = disentangled_boxes.to(torch.float32).corners

            loss = smooth_l1_loss(pred_corners, target_corners, beta=self.smooth_l1_loss_beta)

            # Bound the loss
            loss.clamp(max=self.max_loss_per_group)

            if weights is not None:
                # loss = torch.sum(loss.reshape(-1, 24) * weights.unsqueeze(-1))
                loss = torch.sum(loss.reshape(-1, 24).mean(dim=1) * weights)
            else:
                loss = loss.reshape(-1, 24).mean()

            disentangled_losses["loss_box3d_" + component_key] = loss

        entangled_l1_dist = (target_corners - box3d_pred.corners).detach().abs().reshape(-1, 24).mean(dim=1)

        return disentangled_losses, entangled_l1_dist
