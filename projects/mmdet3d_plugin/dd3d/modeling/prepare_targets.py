# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch

from detectron2.layers import cat

from projects.mmdet3d_plugin.dd3d.structures.boxes3d import Boxes3D

INF = 100000000.


class DD3DTargetPreparer():
    def __init__(self, 
                 num_classes, 
                 input_shape,
                 box3d_on=True,
                 center_sample=True,
                 pos_radius=1.5,
                 sizes_of_interest=None):
        self.num_classes = num_classes
        self.center_sample = center_sample
        self.strides = [shape.stride for shape in input_shape]
        self.radius = pos_radius
        self.dd3d_enabled = box3d_on

        # generate sizes of interest
        # NOTE:
        # soi = []
        # prev_size = -1
        # for s in sizes_of_interest:
        #     soi.append([prev_size, s])
        #     prev_size = s
        # soi.append([prev_size, INF])
        self.sizes_of_interest = sizes_of_interest

    def __call__(self, locations, gt_instances, feature_shapes):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(loc_to_size_range_per_level[None].expand(num_loc_list[l], -1))

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(locations, gt_instances, loc_to_size_range, num_loc_list)

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        box2d = training_targets.pop("box2d", None)

        # transpose im first training_targets to level first ones
        training_targets = {k: self._transpose(v, num_loc_list) for k, v in training_targets.items() if k != "box2d"}

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level for level, loc in enumerate(training_targets["locations"])
        ]

        # Flatten targets: (L x B x H x W, TARGET_SIZE)
        labels = cat([x.reshape(-1) for x in training_targets["labels"]])
        box2d_reg_targets = cat([x.reshape(-1, 4) for x in training_targets["box2d_reg"]])

        target_inds = cat([x.reshape(-1) for x in training_targets["target_inds"]])
        locations = cat([x.reshape(-1, 2) for x in training_targets["locations"]])
        im_inds = cat([x.reshape(-1) for x in training_targets["im_inds"]])
        fpn_levels = cat([x.reshape(-1) for x in training_targets["fpn_levels"]])

        pos_inds = torch.nonzero(labels != self.num_classes).squeeze(1)

        targets = {
            "labels": labels,
            "box2d_reg_targets": box2d_reg_targets,
            "locations": locations,
            "target_inds": target_inds,
            "im_inds": im_inds,
            "fpn_levels": fpn_levels,
            "pos_inds": pos_inds
        }

        if self.dd3d_enabled:
            box3d_targets = Boxes3D.cat(training_targets["box3d"])
            targets.update({"box3d_targets": box3d_targets})

            if box2d is not None:
                # Original format is B x L x (H x W, 4)
                # Need to be in L x (B, 4, H, W).
                batched_box2d = []
                for lvl, per_lvl_box2d in enumerate(zip(*box2d)):
                    # B x (H x W, 4)
                    h, w = feature_shapes[lvl]
                    batched_box2d_lvl = torch.stack([x.T.reshape(4, h, w) for x in per_lvl_box2d], dim=0)
                    batched_box2d.append(batched_box2d_lvl)
                targets.update({"batched_box2d": batched_box2d})

        return targets

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        box2d_reg = []

        if self.dd3d_enabled:
            box3d = []

        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                # reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                box2d_reg.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)

                if self.dd3d_enabled:
                    box3d.append(
                        Boxes3D(
                            locations.new_zeros(locations.size(0), 4),
                            locations.new_zeros(locations.size(0), 2),
                            locations.new_zeros(locations.size(0), 1),
                            locations.new_zeros(locations.size(0), 3),
                            locations.new_zeros(locations.size(0), 3, 3),
                        ).to(torch.float32)
                    )
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            # reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            box2d_reg_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(bboxes, num_loc_list, xs, ys)
            else:
                is_in_boxes = box2d_reg_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = box2d_reg_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            box2d_reg_per_im = box2d_reg_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            box2d_reg.append(box2d_reg_per_im)
            target_inds.append(target_inds_per_im)

            if self.dd3d_enabled:
                # 3D box targets
                box3d_per_im = targets_per_im.gt_boxes3d[locations_to_gt_inds]
                box3d.append(box3d_per_im)

        ret = {"labels": labels, "box2d_reg": box2d_reg, "target_inds": target_inds}
        if self.dd3d_enabled:
            ret.update({"box3d": box3d})

        return ret

    def get_sample_region(self, boxes, num_loc_list, loc_xs, loc_ys):
        center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
        center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = self.strides[level] * self.radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        if isinstance(training_targets[0], Boxes3D):
            for im_i in range(len(training_targets)):
                # training_targets[im_i] = torch.split(training_targets[im_i], num_loc_list, dim=0)
                training_targets[im_i] = training_targets[im_i].split(num_loc_list, dim=0)

            targets_level_first = []
            for targets_per_level in zip(*training_targets):
                targets_level_first.append(Boxes3D.cat(targets_per_level, dim=0))
            return targets_level_first

        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(training_targets[im_i], num_loc_list, dim=0)

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(torch.cat(targets_per_level, dim=0))
        return targets_level_first
