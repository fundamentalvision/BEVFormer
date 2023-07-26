# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch
import torch.nn.functional as F


def compute_features_locations(h, w, stride, dtype=torch.float32, device='cpu', offset="none"):
    """Adapted from AdelaiDet:
        https://github.com/aim-uofa/AdelaiDet/blob/master/adet/utils/comm.py

    Key differnece: offset is configurable.
    """
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=dtype, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=dtype, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # (dennis.park)
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    locations = torch.stack((shift_x, shift_y), dim=1)
    if offset == "half":
        locations += stride // 2
    else:
        assert offset == "none"

    return locations


def aligned_bilinear(tensor, factor, offset="none"):
    """Adapted from AdelaiDet:
        https://github.com/aim-uofa/AdelaiDet/blob/master/adet/utils/comm.py
    """
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor, size=(oh, ow), mode='bilinear', align_corners=True)
    if offset == "half":
        tensor = F.pad(tensor, pad=(factor // 2, 0, factor // 2, 0), mode="replicate")

    return tensor[:, :, :oh - 1, :ow - 1]
