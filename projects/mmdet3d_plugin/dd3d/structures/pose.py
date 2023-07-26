# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np
from pyquaternion import Quaternion


class Pose:
    """SE(3) rigid transform class that allows compounding of 6-DOF poses
    and provides common transformations that are commonly seen in geometric problems.
    """
    def __init__(self, wxyz=np.float32([1., 0., 0., 0.]), tvec=np.float32([0., 0., 0.])):
        """Initialize a Pose with Quaternion and 3D Position

        Parameters
        ----------
        wxyz: np.float32 or Quaternion (default: np.float32([1,0,0,0]))
            Quaternion/Rotation (wxyz)

        tvec: np.float32 (default: np.float32([0,0,0]))
            Translation (xyz)
        """
        assert isinstance(wxyz, (np.ndarray, Quaternion))
        assert isinstance(tvec, np.ndarray)

        if isinstance(wxyz, np.ndarray):
            assert np.abs(1.0 - np.linalg.norm(wxyz)) < 1.0e-3

        self.quat = Quaternion(wxyz)
        self.tvec = tvec

    def __repr__(self):
        formatter = {'float_kind': lambda x: '%.2f' % x}
        tvec_str = np.array2string(self.tvec, formatter=formatter)
        return 'wxyz: {}, tvec: ({})'.format(self.quat, tvec_str)

    def copy(self):
        """Return a copy of this pose object.

        Returns
        ----------
        result: Pose
            Copied pose object.
        """
        return self.__class__(Quaternion(self.quat), self.tvec.copy())

    def __mul__(self, other):
        """Left-multiply Pose with another Pose or 3D-Points.

        Parameters
        ----------
        other: Pose or np.ndarray
            1. Pose: Identical to oplus operation.
               (i.e. self_pose * other_pose)
            2. ndarray: transform [N x 3] point set
               (i.e. X' = self_pose * X)

        Returns
        ----------
        result: Pose or np.ndarray
            Transformed pose or point cloud
        """
        if isinstance(other, Pose):
            assert isinstance(other, self.__class__)
            t = self.quat.rotate(other.tvec) + self.tvec
            q = self.quat * other.quat
            return self.__class__(q, t)
        elif isinstance(other, np.ndarray):
            assert other.shape[-1] == 3, 'Point cloud is not 3-dimensional'
            X = np.hstack([other, np.ones((len(other), 1))]).T
            return (np.dot(self.matrix, X).T)[:, :3]
        else:
            return NotImplemented

    def __rmul__(self, other):
        raise NotImplementedError('Right multiply not implemented yet!')

    def inverse(self):
        """Returns a new Pose that corresponds to the
        inverse of this one.

        Returns
        ----------
        result: Pose
            Inverted pose
        """
        qinv = self.quat.inverse
        return self.__class__(qinv, qinv.rotate(-self.tvec))

    @property
    def matrix(self):
        """Returns a 4x4 homogeneous matrix of the form [R t; 0 1]

        Returns
        ----------
        result: np.ndarray
            4x4 homogeneous matrix
        """
        result = self.quat.transformation_matrix
        result[:3, 3] = self.tvec
        return result

    @property
    def rotation_matrix(self):
        """Returns the 3x3 rotation matrix (R)

        Returns
        ----------
        result: np.ndarray
            3x3 rotation matrix
        """
        result = self.quat.transformation_matrix
        return result[:3, :3]

    @property
    def rotation(self):
        """Return the rotation component of the pose as a Quaternion object.

        Returns
        ----------
        self.quat: Quaternion
            Rotation component of the Pose object.
        """
        return self.quat

    @property
    def translation(self):
        """Return the translation component of the pose as a np.ndarray.

        Returns
        ----------
        self.tvec: np.ndarray
            Translation component of the Pose object.
        """
        return self.tvec

    @classmethod
    def from_matrix(cls, transformation_matrix):
        """Initialize pose from 4x4 transformation matrix

        Parameters
        ----------
        transformation_matrix: np.ndarray
            4x4 containing rotation/translation

        Returns
        -------
        Pose
        """
        return cls(wxyz=Quaternion(matrix=transformation_matrix[:3, :3]), tvec=np.float32(transformation_matrix[:3, 3]))

    @classmethod
    def from_rotation_translation(cls, rotation_matrix, tvec):
        """Initialize pose from rotation matrix and translation vector.

        Parameters
        ----------
        rotation_matrix : np.ndarray
            3x3 rotation matrix
        tvec : np.ndarray
            length-3 translation vector
        """
        return cls(wxyz=Quaternion(matrix=rotation_matrix), tvec=np.float64(tvec))

    def __eq__(self, other):
        return self.quat == other.quat and (self.tvec == other.tvec).all()
