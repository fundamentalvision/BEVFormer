# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np
import torch
from pyquaternion import Quaternion
from torch.cuda import amp

from projects.mmdet3d_plugin.dd3d.utils.geometry import unproject_points2d
import projects.mmdet3d_plugin.dd3d.structures.transform3d as t3d
# yapf: disable
BOX3D_CORNER_MAPPING = [
    [1, 1, 1, 1, -1, -1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1]
]
# yapf: enable

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

def _to_tensor(x, dim):
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32)
    elif isinstance(x, np.ndarray) or isinstance(x, list) or isinstance(x, tuple):
        x = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, Quaternion):
        x = torch.tensor(x.elements, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported type: {type(x).__name__}")

    if x.ndim == 1:
        x = x.reshape(-1, dim)
    elif x.ndim > 2:
        raise ValueError(f"Invalid shape of input: {x.shape.__str__()}")
    return x


class GenericBoxes3D():
    def __init__(self, quat, tvec, size):
        self.quat = _to_tensor(quat, dim=4)
        self._tvec = _to_tensor(tvec, dim=3)
        self.size = _to_tensor(size, dim=3)

    @property
    def tvec(self):
        return self._tvec

    @property
    @amp.autocast(enabled=False)
    def corners(self):
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        translation = t3d.Translate(self.tvec, device=self.device)

        R = quaternion_to_matrix(self.quat)
        rotation = t3d.Rotate(R=R.transpose(1, 2), device=self.device)  # Need to transpose to make it work.

        tfm = rotation.compose(translation)

        _corners = 0.5 * self.quat.new_tensor(BOX3D_CORNER_MAPPING).T
        # corners_in_obj_frame = self.size.unsqueeze(1) * _corners.unsqueeze(0)
        lwh = self.size[:, [1, 0, 2]]  # wlh -> lwh
        corners_in_obj_frame = lwh.unsqueeze(1) * _corners.unsqueeze(0)

        corners3d = tfm.transform_points(corners_in_obj_frame)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        return corners3d

    @classmethod
    def from_vectors(cls, vecs, device="cpu"):
        """
        Parameters
        ----------
        vecs: Iterable[np.ndarray]
            Iterable of 10D pose representation.

        intrinsics: np.ndarray
            (3, 3) intrinsics matrix.
        """
        quats, tvecs, sizes = [], [], []
        for vec in vecs:
            quat = vec[:4]
            tvec = vec[4:7]
            size = vec[7:]

            quats.append(quat)
            tvecs.append(tvec)
            sizes.append(size)

        quats = torch.as_tensor(quats, dtype=torch.float32, device=device)
        tvecs = torch.as_tensor(tvecs, dtype=torch.float32, device=device)
        sizes = torch.as_tensor(sizes, device=device)

        return cls(quats, tvecs, sizes)

    @classmethod
    def cat(cls, boxes_list, dim=0):

        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0), torch.empty(0), torch.empty(0))
        assert all([isinstance(box, GenericBoxes3D) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        quat = torch.cat([b.quat for b in boxes_list], dim=dim)
        tvec = torch.cat([b.tvec for b in boxes_list], dim=dim)
        size = torch.cat([b.size for b in boxes_list], dim=dim)

        cat_boxes = cls(quat, tvec, size)
        return cat_boxes

    def split(self, split_sizes, dim=0):
        assert sum(split_sizes) == len(self)
        quat_list = torch.split(self.quat, split_sizes, dim=dim)
        tvec_list = torch.split(self.tvec, split_sizes, dim=dim)
        size_list = torch.split(self.size, split_sizes, dim=dim)

        return [GenericBoxes3D(*x) for x in zip(quat_list, tvec_list, size_list)]

    def __getitem__(self, item):
        """
        """
        if isinstance(item, int):
            return GenericBoxes3D(self.quat[item].view(1, -1), self.tvec[item].view(1, -1), self.size[item].view(1, -1))

        quat = self.quat[item]
        tvec = self.tvec[item]
        size = self.size[item]

        assert quat.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert tvec.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert size.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)

        return GenericBoxes3D(quat, tvec, size)

    def __len__(self):
        assert len(self.quat) == len(self.tvec) == len(self.size)
        return self.quat.shape[0]

    def clone(self):
        """
        """
        return GenericBoxes3D(self.quat.clone(), self.tvec.clone(), self.size.clone())

    def vectorize(self):
        xyz = self.tvec
        return torch.cat([self.quat, xyz, self.size], dim=1)

    @property
    def device(self):
        return self.quat.device

    def to(self, *args, **kwargs):
        quat = self.quat.to(*args, **kwargs)
        tvec = self.tvec.to(*args, **kwargs)
        size = self.size.to(*args, **kwargs)
        return GenericBoxes3D(quat, tvec, size)


class Boxes3D(GenericBoxes3D):
    """Vision-based 3D box container.

    The tvec is computed from projected center, depth, and intrinsics.
    """
    def __init__(self, quat, proj_ctr, depth, size, inv_intrinsics):
        self.quat = quat
        self.proj_ctr = proj_ctr
        self.depth = depth
        self.size = size
        self.inv_intrinsics = inv_intrinsics

    @property
    def tvec(self):
        ray = unproject_points2d(self.proj_ctr, self.inv_intrinsics)
        xyz = ray * self.depth
        return xyz

    @classmethod
    def from_vectors(cls, vecs, intrinsics, device="cpu"):
        """
        Parameters
        ----------
        vecs: Iterable[np.ndarray]
            Iterable of 10D pose representation.

        intrinsics: np.ndarray
            (3, 3) intrinsics matrix.
        """
        if len(vecs) == 0:
            quats = torch.as_tensor([], dtype=torch.float32, device=device).view(-1, 4)
            proj_ctrs = torch.as_tensor([], dtype=torch.float32, device=device).view(-1, 2)
            depths = torch.as_tensor([], dtype=torch.float32, device=device).view(-1, 1)
            sizes = torch.as_tensor([], dtype=torch.float32, device=device).view(-1, 3)
            inv_intrinsics = torch.as_tensor([], dtype=torch.float32, device=device).view(-1, 3, 3)
            return cls(quats, proj_ctrs, depths, sizes, inv_intrinsics)

        quats, proj_ctrs, depths, sizes = [], [], [], []
        for vec in vecs:
            quat = vec[:4]

            proj_ctr = intrinsics.dot(vec[4:7])
            proj_ctr = proj_ctr[:2] / proj_ctr[-1]

            depth = vec[6:7]

            size = vec[7:]

            quats.append(quat)
            proj_ctrs.append(proj_ctr)
            depths.append(depth)
            sizes.append(size)

        quats = torch.as_tensor(np.array(quats), dtype=torch.float32, device=device)
        proj_ctrs = torch.as_tensor(np.array(proj_ctrs), dtype=torch.float32, device=device)
        depths = torch.as_tensor(np.array(depths), dtype=torch.float32, device=device)
        sizes = torch.as_tensor(np.array(sizes), dtype=torch.float32, device=device)

        inv_intrinsics = np.linalg.inv(intrinsics)
        inv_intrinsics = torch.as_tensor(inv_intrinsics[None, ...], device=device).expand(len(vecs), 3, 3)

        return cls(quats, proj_ctrs, depths, sizes, inv_intrinsics)

    @classmethod
    def cat(cls, boxes_list, dim=0):

        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
        assert all([isinstance(box, Boxes3D) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        quat = torch.cat([b.quat for b in boxes_list], dim=dim)
        proj_ctr = torch.cat([b.proj_ctr for b in boxes_list], dim=dim)
        depth = torch.cat([b.depth for b in boxes_list], dim=dim)
        size = torch.cat([b.size for b in boxes_list], dim=dim)
        inv_intrinsics = torch.cat([b.inv_intrinsics for b in boxes_list], dim=dim)

        cat_boxes = cls(quat, proj_ctr, depth, size, inv_intrinsics)
        return cat_boxes

    def split(self, split_sizes, dim=0):
        assert sum(split_sizes) == len(self)
        quat_list = torch.split(self.quat, split_sizes, dim=dim)
        proj_ctr_list = torch.split(self.proj_ctr, split_sizes, dim=dim)
        depth_list = torch.split(self.depth, split_sizes, dim=dim)
        size_list = torch.split(self.size, split_sizes, dim=dim)
        inv_K_list = torch.split(self.inv_intrinsics, split_sizes, dim=dim)

        return [Boxes3D(*x) for x in zip(quat_list, proj_ctr_list, depth_list, size_list, inv_K_list)]

    def __getitem__(self, item):
        """
        """
        if isinstance(item, int):
            return Boxes3D(
                self.quat[item].view(1, -1), self.proj_ctr[item].view(1, -1), self.depth[item].view(1, -1),
                self.size[item].view(1, -1), self.inv_intrinsics[item].view(1, 3, 3)
            )

        quat = self.quat[item]
        ctr = self.proj_ctr[item]
        depth = self.depth[item]
        size = self.size[item]
        inv_K = self.inv_intrinsics[item]

        assert quat.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert ctr.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert depth.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert size.dim() == 2, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert inv_K.dim() == 3, "Indexing on Boxes3D with {} failed to return a matrix!".format(item)
        assert inv_K.shape[1:] == (3, 3), "Indexing on Boxes3D with {} failed to return a matrix!".format(item)

        return Boxes3D(quat, ctr, depth, size, inv_K)

    def __len__(self):
        assert len(self.quat) == len(self.proj_ctr) == len(self.depth) == len(self.size) == len(self.inv_intrinsics)
        return self.quat.shape[0]

    def clone(self):
        """
        """
        return Boxes3D(
            self.quat.clone(), self.proj_ctr.clone(), self.depth.clone(), self.size.clone(), self.inv_intrinsics.clone()
        )

    def to(self, *args, **kwargs):
        quat = self.quat.to(*args, **kwargs)
        proj_ctr = self.proj_ctr.to(*args, **kwargs)
        depth = self.depth.to(*args, **kwargs)
        size = self.size.to(*args, **kwargs)
        inv_K = self.inv_intrinsics.to(*args, **kwargs)
        return Boxes3D(quat, proj_ctr, depth, size, inv_K)
