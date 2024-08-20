from ...Transformations import Transform3d, Transform2d
from ...Matrices.matrix4 import Matrix4
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np


DRAW_NORMALS = 1
DRAW_GIZMOS  = 2
DRAW_BASIS   = 4


class Shape:
    def __init__(self):
        self._transform = Transform3d()
        self._uv_transform = Transform2d()
        self.resolution = (32, 32)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        self._resolution = tuple(int(val) for idx, val in zip(range(2), value))

    def surface_orientation(self) -> float:
        return 0.0

    @property
    def start_point(self) -> Vector3:
        return 0.5 * (self.point(Vector2(0.0, 0.0)) + self.point(Vector2(0.0, 0.5)))

    @property
    def end_point(self) -> Vector3:
        return 0.5 * (self.point(Vector2(1.0, 0.0)) + self.point(Vector2(1.0, 0.5)))

    @property
    def transform(self) -> Transform3d:
        return self._transform

    @property
    def uv_transform(self) -> Transform2d:
        return self._uv_transform

    def world_space_normal(self, uv: Vector2) -> Vector3:
        return self.transform.transform_direction(self.normal(uv))

    def world_space_point(self, uv: Vector2) -> Vector3:
        return self.transform.transform_point(self.point(uv))

    def normal(self, uv: Vector2) -> Vector3:
        u1, u2 = min(max(0.0, uv.x + 1e-5), 1.0), min(max(0.0, uv.x - 1e-5), 1.0)
        v1, v2 = min(max(0.0, uv.y + 1e-5), 1.0), min(max(0.0, uv.y - 1e-5), 1.0)
        dpu = (self.point(Vector2(u1, uv.y)) - self.point(Vector2(u2, uv.y))).normalize()
        dpv = (self.point(Vector2(uv.x, v1)) - self.point(Vector2(uv.x, v2))).normalize()
        return Vector3.cross(dpu, dpv).normalize()

    def point(self, uv: Vector2) -> Vector3:
        return Vector3(0.0, 0.0, 0.0)

    @property
    def basis_start(self) -> Matrix4:
        o = self.start_point
        r = (self.point(Vector2(0.0, 0.0)) - o).normalize()
        u = (self.point(Vector2(0.0, 0.25)) - o).normalize()
        f = Vector3.cross(r, u)
        return self.transform.transform_matrix * Matrix4.build_transform(r, f, u, o)

    @property
    def basis_end(self) -> Matrix4:
        o = self.end_point
        r = (self.point(Vector2(1.0, 0.0)) - o).normalize()
        u = (self.point(Vector2(1.0, 0.25)) - o).normalize()
        f = Vector3.cross(r, u)
        return self.transform.transform_matrix * Matrix4.build_transform(r, f, u, o)

    def build_shape_points(self) -> Tuple[np.ndarray, ...]:
        us, vs = np.linspace(0.0, 1.0, self.resolution[0]), np.linspace(0.0, 1.0, self.resolution[1])
        xs = []
        ys = []
        zs = []
        for u in us:
            x_row = []
            y_row = []
            z_row = []
            xs.append(x_row)
            ys.append(y_row)
            zs.append(z_row)
            for v in vs:
                x, y, z = self.world_space_point(Vector2(u, v))
                x_row.append(x)
                y_row.append(y)
                z_row.append(z)
        return np.array(xs), np.array(ys), np.array(zs)

    def draw_shape_basis(self, axis=None, axis_length: float = 0.125):
        axis = axis if axis else plt.axes(projection='3d')
        basis = self.transform
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

    def draw_shape_normals(self, axis=None, normal_length: float = 0.125):
        axis = axis if axis else plt.axes(projection='3d')
        normals_per_length, normals_per_diam = self.resolution
        for index in range(normals_per_length * normals_per_diam):
            row, col = divmod(index, normals_per_diam)
            uv = Vector2(row / (normals_per_length - 1), col / (normals_per_diam - 1))
            hp = self.world_space_point(uv)
            hn = self.world_space_normal(uv)
            axis.plot((hp.x, hp.x + hn.x * normal_length),
                      (hp.y, hp.y + hn.y * normal_length),
                      (hp.z, hp.z + hn.z * normal_length), 'r')
        return axis

    def draw_shape_gizmos(self, axis=None):
        return axis

    def draw_shape(self, axis=None, args_mask: int = 0, show: bool = False, **kwargs):
        axis = axis if axis else plt.axes(projection='3d')
        shape = self.build_shape_points()
        if len(kwargs) != 0:
            args = {k: v for k, v in kwargs.items() if k in {"linewidths", "antialiased", "color", "edgecolor", "alpha"}}
        else:
            args = {"linewidths": 0.0, "antialiased": True, "color": "white", "edgecolor": "none", "alpha": 1.0}
        axis.plot_surface(*shape, **args)
        axis.set_xlabel("x, [mm]")
        axis.set_ylabel("y, [mm]")
        axis.set_zlabel("z, [mm]")
        if args_mask & DRAW_BASIS:
            self.draw_shape_basis(axis)
        if args_mask & DRAW_NORMALS:
            self.draw_shape_normals(axis)
        if args_mask & DRAW_GIZMOS:
            self.draw_shape_gizmos(axis)
        if show:
            axis.set_aspect('equal', 'box')
            plt.show()
        return axis

