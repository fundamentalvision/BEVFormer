# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)

PI = 3.14159265358979323846
EPS = 1e-7

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def allocentric_to_egocentric(quat, proj_ctr, inv_intrinsics):
    """
    Parameters
    ----------
    quat: Tensor
        (N, 4). Batch of (allocentric) quaternions.

    proj_ctr: Tensor
        (N, 2). Projected centers. xy coordninates.

    inv_intrinsics: [type]
        (N, 3, 3). Inverted intrinsics.
    """
    R_obj_to_local = quaternion_to_matrix(quat)

    # ray == z-axis in local orientaion
    ray = unproject_points2d(proj_ctr, inv_intrinsics)
    z = ray / ray.norm(dim=1, keepdim=True)

    # gram-schmit process: local_y = global_y - global_y \dot local_z
    y = z.new_tensor([[0., 1., 0.]]) - z[:, 1:2] * z
    y = y / y.norm(dim=1, keepdim=True)
    x = torch.cross(y, z, dim=1)

    # local -> global
    R_local_to_global = torch.stack([x, y, z], dim=-1)

    # obj -> global
    R_obj_to_global = torch.bmm(R_local_to_global, R_obj_to_local)

    egocentric_quat = matrix_to_quaternion(R_obj_to_global)

    # Make sure it's unit norm.
    quat_norm = egocentric_quat.norm(dim=1, keepdim=True)
    if not torch.allclose(quat_norm, torch.as_tensor(1.), atol=1e-3):
        LOG.warning(
            f"Some of the input quaternions are not unit norm: min={quat_norm.min()}, max={quat_norm.max()}; therefore normalizing."
        )
        egocentric_quat = egocentric_quat / quat_norm.clamp(min=EPS)

    return egocentric_quat


def homogenize_points(xy):
    """
    Parameters
    ----------
    xy: Tensor
        xy coordinates. shape=(N, ..., 2)
        E.g., (N, 2) or (N, K, 2) or (N, H, W, 2)

    Returns
    -------
    Tensor:
        1. is appended to the last dimension. shape=(N, ..., 3)
        E.g, (N, 3) or (N, K, 3) or (N, H, W, 3).
    """
    # NOTE: this seems to work for arbitrary number of dimensions of input
    pad = torch.nn.ConstantPad1d(padding=(0, 1), value=1.)
    return pad(xy)


def project_points3d(Xw, K):
    _, C = Xw.shape
    assert C == 3
    uv, _ = cv2.projectPoints(
        Xw, np.zeros((3, 1), dtype=np.float32), np.zeros(3, dtype=np.float32), K, np.zeros(5, dtype=np.float32)
    )
    return uv.reshape(-1, 2)


def unproject_points2d(points2d, inv_K, scale=1.0):
    """
    Parameters
    ----------
    points2d: Tensor
        xy coordinates. shape=(N, ..., 2)
        E.g., (N, 2) or (N, K, 2) or (N, H, W, 2)

    inv_K: Tensor
        Inverted intrinsics; shape=(N, 3, 3)

    scale: float, default: 1.0
        Scaling factor.

    Returns
    -------
    Tensor:
        Unprojected 3D point. shape=(N, ..., 3)
        E.g., (N, 3) or (N, K, 3) or (N, H, W, 3)
    """
    points2d = homogenize_points(points2d)
    siz = points2d.size()
    points2d = points2d.view(-1, 3).unsqueeze(-1)  # (N, 3, 1)
    unprojected = torch.matmul(inv_K, points2d)  # (N, 3, 3) x (N, 3, 1) -> (N, 3, 1)
    unprojected = unprojected.view(siz)

    return unprojected * scale
