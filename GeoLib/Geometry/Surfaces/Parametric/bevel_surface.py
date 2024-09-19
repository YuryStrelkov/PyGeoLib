from ...Surfaces.Curves.bezier import bezier_interpolate_pt_3d, bezier_interpolate_tangent_3d
from ...Matrices.matrix4 import Matrix4
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from matplotlib import pyplot as plt
from typing import Iterable, List
from .parametric_surface import ParametricSurface
import numpy as np


def linear_interpolate_pt_3d(param_t: float, points: List[Vector3]) -> Vector3:
    t = param_t * ((len(points) ) // 2)
    scet_index = int(t - 1e-6)
    t = t - int(t - 1e-6)
    p1 = points[scet_index * 2]
    p2 = points[scet_index * 2 + 1]
    return Vector3.lerp(p1, p2, t)


def linear_interpolate_tangent_pt_3d(param_t: float, points: List[Vector3]) -> Vector3:
    t = param_t * ((len(points) ) // 2)
    scet_index = int(t - 1e-6)
    p1 = points[scet_index * 2]
    p2 = points[scet_index * 2 + 1]
    return (p2 - p1).normalize()


def _draw_curve(curve: Iterable[Vector3], axis=None, color='b'):
    axis = axis if axis else plt.axes(projection='3d')
    x = tuple(v.x for v in curve)
    y = tuple(v.y for v in curve)
    z = tuple(v.z for v in curve)
    axis.plot(x, y, z, color)
    return axis


def _draw_basis(basis: Matrix4, axis=None, axis_length: float = 0.125):
    axis = axis if axis else plt.axes(projection='3d')
    o = basis.origin
    r = basis.right
    u = basis.up
    f = basis.front
    axis.plot((o.x, o.x + r.x * axis_length),
              (o.y, o.y + r.y * axis_length),
              (o.z, o.z + r.z * axis_length), 'r')

    axis.plot((o.x, o.x + u.x * axis_length),
              (o.y, o.y + u.y * axis_length),
              (o.z, o.z + u.z * axis_length), 'g')

    axis.plot((o.x, o.x + f.x * axis_length),
              (o.y, o.y + f.y * axis_length),
              (o.z, o.z + f.z * axis_length), 'b')
    return axis


class BevelSurface(ParametricSurface):

    DIRECTIONS = (Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0))

    @staticmethod
    def _build_basis(forward: Vector3, origin: Vector3 = None, prev_tbn: Matrix4 = None) -> Matrix4:
        up = prev_tbn.up if prev_tbn else min(BevelSurface.DIRECTIONS, key=lambda v: abs(Vector3.dot(v, forward)))
        right = Vector3.cross(forward, up).normalize()
        up = Vector3.cross(right, forward).normalize()
        return Matrix4.build_transform(right, up, forward, origin)

    def _build_local_tbn_matrices(self) -> None:
        p1, p2 = self._profile_shape[0], self._profile_shape[1]
        forward = (p2 - p1).normalize()
        tbn = BevelSurface._build_basis(forward, p1)
        self._tbn_matrices = [tbn]
        for p1, p2 in zip(self._profile_shape[1:-1], self._profile_shape[2:]):
            forward = (p2 - p1).normalize()
            if forward.magnitude < 1e-6:
                _tbn = tbn
            else:
                _tbn = BevelSurface._build_basis(forward, p1, tbn)
            self._tbn_matrices.append(_tbn)
            tbn = _tbn
        p2, p1 = self._profile_shape[-2], self._profile_shape[-1]
        forward = (p1 - p2).normalize()
        self._tbn_matrices.append(BevelSurface._build_basis(forward, p1, tbn))

    def __init__(self, start_shape: Iterable[Vector3],
                 profile_shape: Iterable[Vector3],
                 end_shape: Iterable[Vector3] = None):
        super().__init__()
        self._start_shape: List[Vector3] = list(v for v in start_shape)
        self._final_shape: List[Vector3] = list(v for v in end_shape) \
            if end_shape else list(v for v in start_shape)
        self._profile_shape: List[Vector3] = list(v for v in profile_shape)
        self.resolution = (32, 512)
        self._interpolate_mode = 1
        self._tbn_matrices = []
        self._build_local_tbn_matrices()

    @property
    def interpolation_mode(self) -> int:
        return self._interpolate_mode

    @interpolation_mode.setter
    def interpolation_mode(self, value: int) -> None:
        self._interpolate_mode = int(value) if int(value) in (0, 1) else self._interpolate_mode

    def draw_shape_gizmos(self, axis=None):
        for tbn in self._tbn_matrices:
            axis = _draw_basis(tbn, axis)
        t1: Matrix4 = self._tbn_matrices[0]
        t2: Matrix4 = self._tbn_matrices[-1]
        ts = np.linspace(0.0, 1.0, 218)
        axis = _draw_curve(tuple(bezier_interpolate_pt_3d(t, self._profile_shape) for t in ts.flat), axis, 'k')
        axis = _draw_curve(tuple(t1.multiply_by_point(p) for p in self._start_shape), axis, ':r')
        axis = _draw_curve(tuple(t2.multiply_by_point(p) for p in self._final_shape), axis, ':g')
        return _draw_curve(self._profile_shape, axis, ':b')

    def _bezier_point(self, uv: Vector2) -> Vector3:
        p1 = bezier_interpolate_pt_3d(uv.y, self._start_shape)
        p2 = bezier_interpolate_pt_3d(uv.y, self._final_shape)
        p3 = bezier_interpolate_pt_3d(uv.x, self._profile_shape)
        tbn_id = int(uv.x * (len(self._tbn_matrices) - 1))
        tbn = self._tbn_matrices[tbn_id]
        front = bezier_interpolate_tangent_3d(uv.x - 2 * 1e-6, self._profile_shape).normalize()
        up = Vector3.cross(tbn.up, front).normalize()
        right = Vector3.cross(front, up).normalize()
        up = Vector3.cross(right, front).normalize()
        return Matrix4.build_transform(right, up, front, p3).multiply_by_point(Vector3.lerp(p1, p2, uv.x))

    def _linear_point(self, uv: Vector2) -> Vector3:
        p1 = linear_interpolate_pt_3d(uv.y, self._start_shape)
        p2 = linear_interpolate_pt_3d(uv.y, self._final_shape)
        p3 = linear_interpolate_pt_3d(uv.x, self._profile_shape)
        tbn_id = int(uv.x * (len(self._tbn_matrices) - 1))
        tbn = self._tbn_matrices[tbn_id]
        front = linear_interpolate_tangent_pt_3d(uv.x - 2 * 1e-6, self._profile_shape).normalize()
        up = Vector3.cross(tbn.up, front).normalize()
        right = Vector3.cross(front, up).normalize()
        up = Vector3.cross(right, front).normalize()
        return Matrix4.build_transform(right, up, front, p3).multiply_by_point(Vector3.lerp(p1, p2, uv.x))

    def point(self, uv: Vector2) -> Vector3:
        if self._interpolate_mode == 0:
            return self._linear_point(uv)
        return self._bezier_point(uv)
