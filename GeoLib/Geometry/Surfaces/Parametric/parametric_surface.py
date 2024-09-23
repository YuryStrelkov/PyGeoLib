from typing import Tuple, Generator, Callable, TypeVar, Iterable
from ...Transformations import Transform3d, Transform2d
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from matplotlib import pyplot as plt
import dataclasses
import numpy as np


DRAW_NORMALS = 1
DRAW_GIZMOS  = 2
DRAW_BASIS   = 4

IteratorType = TypeVar('IteratorType')


@dataclasses.dataclass(frozen=True)
class SurfaceVertex:
    position: Vector3
    normal: Vector3
    uv: Vector2


@dataclasses.dataclass(frozen=True)
class SurfaceVertexExt:
    position: Vector3
    normal: Vector3
    tangent: Vector3
    uv: Vector2


class ParametricSurface:
    @staticmethod
    def _calc_indices(index: int, stride: int, shift: int) -> Tuple[int, ...]:
        row, col = divmod(index, stride - 1)
        p1 = shift + col + row * stride
        p2 = shift + 1 + col + row * stride
        p3 = shift + 1 + col + (row + 1) * stride
        p4 = shift + col + (row + 1) * stride
        return p1, p2, p3, p4

    def __init__(self):
        self._transform = Transform3d()
        self._uv_transform = Transform2d()
        self._resolution = (32, 32)
        self._inner_oriented = True
        self._triangulate = True

    def _obj_format(self, surface_id: int = 0, indices_shift: int = 0) -> str:
        header = f"#\n# object shape{surface_id}\n#"
        vertices = '\n'.join(f"v  {vec.x:4.6} {vec.y:4.6} {vec.z:4.6}" for vec in self.positions)
        n_vertices = f"# {self.vertices_count} vertices"
        normals = '\n'.join(f"vn {vec.x:4.6} {vec.y:4.6} {vec.z:4.6}" for vec in self.normals)
        n_normals = f"# {self.normals_count} normals"
        uvs = '\n'.join(f"vt {vec.x:4.6} {vec.y:4.6}" for vec in self.uvs)
        n_uvs = f"# {self.uvs_count} texture coordinates"
        if self.triangulate:
            faces = '\n'.join(f"f {p1 + 1}/{p1 + 1}/{p1 + 1}"
                              f" {p3 + 1}/{p3 + 1}/{p3 + 1}"
                              f" {p2 + 1}/{p2 + 1}/{p2 + 1}" for p1, p2, p3 in self.triangle_faces(indices_shift))
            n_faces = f"# {self.triangles_count} triangle faces"
        else:
            faces = '\n'.join(f"f {p1}/{p1}/{p1}"
                              f" {p4}/{p4}/{p4}"
                              f" {p3}/{p3}/{p3}"
                              f" {p2}/{p2}/{p2}" for p1, p2, p3, p4 in self.quad_faces(indices_shift))
            n_faces = f"# {self.quades_count} quad faces"
        all_stuff = (header, vertices, n_vertices, normals, n_normals, uvs, n_uvs, faces, n_faces)
        return '\n'.join(stuff for stuff in all_stuff)

    def __str__(self) -> str:
        return f"{self:txt}"

    def __format__(self, format_spec) -> str:
        match format_spec:
            case 'json': return self._obj_format()
            case 'obj': return self._obj_format()
            case 'txt': return self._obj_format()
            case _: return self._obj_format()

    @property
    def vertices_count(self) -> int:
        return self.resolution[0] * self.resolution[1]

    @property
    def uvs_count(self) -> int:
        return self.vertices_count

    @property
    def normals_count(self) -> int:
        return self.vertices_count

    @property
    def quades_count(self) -> int:
        return (self.resolution[0] - 1) * (self.resolution[1] -  1)

    @property
    def triangles_count(self) -> int:
        return 2 * self.quades_count

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        self._resolution = tuple(int(val) for idx, val in zip(range(2), value))

    @property
    def inner_oriented(self) -> bool:
        return self._inner_oriented

    @property
    def triangulate(self) -> bool:
        return self._triangulate

    @inner_oriented.setter
    def inner_oriented(self, value: bool) -> None:
        self._inner_oriented = value

    @triangulate.setter
    def triangulate(self, value: bool) -> None:
        self._triangulate = value

    def surface_orientation(self) -> float:
        return 0.0

    @property
    def transform(self) -> Transform3d:
        return self._transform

    @property
    def uv_transform(self) -> Transform2d:
        return self._uv_transform

    def _iterable_action(self, func: Callable[[Vector2], IteratorType]) -> Generator[IteratorType, None, None]:
        points_per_length, points_per_diam = self.resolution
        n_points = points_per_length * points_per_diam
        for index in range(n_points):
            row, col = divmod(index, points_per_diam)
            uv = Vector2(float(row) / (points_per_length - 1), float(col) / (points_per_diam - 1))
            yield func(uv)

    @property
    def positions(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.point(uv))

    @property
    def normals(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.normal(uv))

    @property
    def tangents(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.tangent(uv))

    @property
    def bi_tangents(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.bi_tangent(uv))

    @property
    def world_space_positions(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.world_space_point(uv))

    @property
    def world_space_normals(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.world_space_normal(uv))

    @property
    def world_space_tangents(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.world_space_tangent(uv))

    @property
    def world_space_bi_tangents(self) -> Generator[Vector3, None, None]:
        return self._iterable_action(lambda uv: self.world_space_bi_tangent(uv))

    @property
    def uvs(self) -> Generator[Vector2, None, None]:
        return self._iterable_action(lambda uv: self._uv_transform.transform_vect(uv, 1.0))

    @property
    def vertices(self) -> Generator[SurfaceVertex, None, None]:
        return self._iterable_action(lambda uv: SurfaceVertex(self.point(uv), self.normal(uv), uv))

    @property
    def vertices_ext(self) -> Generator[SurfaceVertexExt, None, None]:
        return self._iterable_action(lambda uv: SurfaceVertexExt(self.point(uv), self.normal(uv), self.tangent(uv), uv))

    @property
    def world_space_vertices_ext(self) -> Generator[SurfaceVertexExt, None, None]:
        return self._iterable_action(lambda uv: SurfaceVertexExt(
            self.world_space_point(uv), self.world_space_normal(uv), self.world_space_tangent(uv), uv))

    @property
    def world_space_vertices(self) -> Generator[SurfaceVertex, None, None]:
        return self._iterable_action(lambda uv: SurfaceVertex(
            self.world_space_point(uv), self.world_space_normal(uv), uv))

    def world_space_tangent(self, uv: Vector2) -> Vector3:
        return self.transform.transform_direction(self.tangent(uv))

    def world_space_bi_tangent(self, uv: Vector2) -> Vector3:
        return self.transform.transform_direction(self.bi_tangent(uv))

    def world_space_normal(self, uv: Vector2) -> Vector3:
        return self.transform.transform_direction(self.normal(uv))

    def world_space_point(self, uv: Vector2) -> Vector3:
        return self.transform.transform_point(self.point(uv))

    def tangent(self, uv: Vector2) -> Vector3:
        u1, u2 = min(max(0.0, uv.x + 1e-5), 1.0), min(max(0.0, uv.x - 1e-5), 1.0)
        return (self.point(Vector2(u1, uv.y)) - self.point(Vector2(u2, uv.y))).normalize()

    def bi_tangent(self, uv: Vector2) -> Vector3:
        v1, v2 = min(max(0.0, uv.y + 1e-5), 1.0), min(max(0.0, uv.y - 1e-5), 1.0)
        return (self.point(Vector2(uv.x, v1)) - self.point(Vector2(uv.x, v2))).normalize()

    def normal(self, uv: Vector2) -> Vector3:
        return Vector3.cross(self.tangent(uv), self.bi_tangent(uv)).normalize()

    def triangle_faces(self, indices_shift: int = 0) -> Generator[Tuple[int, int, int], None, None]:
        points_per_length, points_per_diam = self.resolution
        n_points = (points_per_length - 1) * (points_per_diam - 1)
        if self.inner_oriented:
            for index in range(n_points):
                p1, p2, p3, p4 = ParametricSurface._calc_indices(index, points_per_diam, indices_shift)
                yield p1, p3, p2
                yield p1, p4, p3
        else:
            for index in range(n_points):
                p1, p2, p3, p4 = ParametricSurface._calc_indices(index, points_per_diam, indices_shift)
                yield p1, p2, p3
                yield p1, p3, p4

    def quad_faces(self, indices_shift: int = 0) -> Generator[Tuple[int, int, int, int], None, None]:
        points_per_length, points_per_diam = self.resolution
        n_points = (points_per_length - 1) * (points_per_diam - 1)
        if self.inner_oriented:
            for index in range(n_points):
                p1, p2, p3, p4 = ParametricSurface._calc_indices(index, points_per_diam, indices_shift)
                yield p1, p4, p3, p2
        else:
            for index in range(n_points):
                p1, p2, p3, p4 = ParametricSurface._calc_indices(index, points_per_diam, indices_shift)
                yield p1, p2, p4, p3

    @property
    def triangles(self) ->  Generator[Tuple[int, int, int], None, None]:
        return self.triangle_faces()

    @property
    def quades(self) ->  Generator[Tuple[int, int, int, int], None, None]:
        return self.quad_faces()

    @property
    def faces_array(self) -> np.ndarray:
        return np.stack(tuple(f for f in self.triangles), dtype=int)

    @property
    def vertices_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.positions), dtype=float)

    @property
    def normals_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.normals), dtype=float)

    @property
    def tangents_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.tangents), dtype=float)

    @property
    def bi_tangents_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.bi_tangents), dtype=float)

    @property
    def world_space_vertices_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.world_space_positions), dtype=float)

    @property
    def world_space_normals_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.world_space_normals), dtype=float)

    @property
    def world_space_tangents_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.world_space_tangents), dtype=float)

    @property
    def world_space_bi_tangents_array(self) -> np.ndarray:
        return np.stack(tuple((p.x, p.y, p.z) for p in self.world_space_bi_tangents), dtype=float)

    @staticmethod
    def save_as_obj(file_name: str, surfaces: Iterable['ParametricSurface']):
        indices_shift = 0
        with open(file_name, 'wt') as out_file:
            for surface_id, surface in enumerate(surfaces):
                print("#", file=out_file)
                print(f"# object shape{surface_id}", file=out_file)
                print("#", file=out_file)

                for pos in surface.positions:
                    print(f"v  {pos.x:4.6} {pos.y:4.6} {pos.z:4.6}", file=out_file)
                print(f"# {surface.vertices_count} vertices", file=out_file)

                for nrm in surface.normals:
                    print(f"vn {nrm.x:4.6} {nrm.y:4.6} {nrm.z:4.6}", file=out_file)
                print(f"# {surface.normals_count} vertices", file=out_file)

                for tex in surface.uvs:
                    print(f"vt {tex.x:4.6} {tex.y:4.6}", file=out_file)
                print(f"# {surface.uvs_count} texture cords", file=out_file)

                print(f"g shape{surface_id}", file=out_file)
                if surface.triangulate:
                    for p1, p2, p3 in surface.triangle_faces(indices_shift):
                        p1, p2, p3 =  p1 + 1, p2 + 1, p3 + 1
                        print(f"f {p1}/{p1}/{p1} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=out_file)
                    print(f"# {surface.triangles_count} triangle faces", file=out_file)
                else:
                    for p1, p2, p3, p4 in surface.quad_faces(indices_shift):
                        p1, p2, p3, p4 =  p1 + 1, p2 + 1, p3 + 1, p4 + 1
                        print(f"f {p1}/{p1}/{p1} {p4}/{p4}/{p4} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=out_file)
                    print(f"# {surface.quades_count} quad faces", file=out_file)
                indices_shift += surface.vertices_count

    def point(self, uv: Vector2) -> Vector3:
        return Vector3(uv.x, 0.0, uv.y)

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

    def draw_shape(self, axis=None, args_mask: int = 0, show: bool = False):
        axis = axis if axis else plt.axes(projection='3d')
        faces = self.faces_array
        vertices = self.world_space_vertices_array
        axis.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, shade=True)
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

