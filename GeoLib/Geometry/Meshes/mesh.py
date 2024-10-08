import numpy as np
from ..Bounds import BoundingBox
from ..common import TWO_PI
from ..Surfaces.Curves import triangulate_polygon
from ..Surfaces import BevelSurface, ParametricSurface, LatheSurface, CylinderSurface
from ..Vectors import Vector3, Vector2
from matplotlib import pyplot as plt
from typing import Union, Tuple, List, Dict
from ..Matrices import Matrix4
import math

LATHE_CUP_START = 1
LATHE_CUP_END = 2
LATHE_CUP_SIDES = LATHE_CUP_START | LATHE_CUP_END
LATHE_CUP_INTERNAL = 4
LATHE_CUP = LATHE_CUP_SIDES | LATHE_CUP_INTERNAL
BEVEL_CUP_START = 8
BEVEL_CUP_END = 16
BEVEL_CUP = BEVEL_CUP_START | BEVEL_CUP_END
MESH_TRIANGULATE = 32


def unique_edges(faces: Tuple[Tuple[Tuple[int, ...], ...], ...]) -> Tuple[Tuple[int, int], ...]:
    edges_dict: Dict[Tuple[int, int], int] = {}
    for (f1, _, _), (f2, _, _), (f3, _, _) in faces:
        for v1, v2 in zip((f1, f2, f3), (f2, f3, f1)):
            pair = (v1, v2) if v1 < v2 else (v2, v1)
            if pair in edges_dict:
                edges_dict[pair] += 1
            else:
                edges_dict.update({pair: 1})
    return tuple(edge for edge, count in edges_dict.items() if count == 1)


class Mesh:
    __slots__ = ('_name', 'vertices', 'uvs', 'faces', 'normals', '_triangulated', 'bounds')

    def __init__(self):
        self._name = f"mesh_{id(self)}"
        self._triangulated = True
        self.vertices: Union[Tuple[Vector3], None] = None
        self.faces: Union[Tuple[Tuple[Tuple[int, ...], ...]], None] = None
        self.uvs: Union[Tuple[Vector3], None] = None
        self.normals: Union[Tuple[Vector3], None] = None
        self.bounds: Union[BoundingBox, None] = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if isinstance(value, str):
            self._name = value

    @property
    def triangulated(self) -> bool:
        return self._triangulated

    @triangulated.setter
    def triangulated(self, value: bool) -> None:
        if isinstance(value, bool):
            self._triangulated = value

    @property
    def border(self) -> Tuple[Tuple[Vector3, Vector3], ...]:
        edges = unique_edges(self.faces)
        return tuple((self.vertices[p1 - 1], self.vertices[p2 - 1]) for p1, p2 in edges)

    @property
    def vertices_array(self) -> np.ndarray:
        return np.array(tuple((v.x, v.y, v.z) for v in self.vertices), dtype=float)

    @property
    def faces_array(self) -> np.ndarray:
        return np.array(tuple((f1[0], f2[0], f3[0]) for f1, f2, f3 in self.faces), dtype=int)

    def compute_uvs(self) -> None:
        if self.vertices is None:
            return
        if len(self.vertices) == 0:
            return
        center = sum(p for p in self.vertices) / len(self.vertices)
        uvs = []
        for p in self.vertices:
            p -= center
            uvs.append(Vector2(math.atan2(p.x, p.y), math.atan2(p.x, p.z)))
        self.uvs = tuple(uvs)

    def compute_normals(self) -> None:
        if self.faces is None:
            return
        if len(self.faces) == 0:
            return
        pt_per_normal = {}
        for (p1, p2, p3) in self.faces:
            v1, v2, v3 = self.vertices[p1], self.vertices[p2], self.vertices[p3]
            normal = Vector3.cross(v1 - v3, v2 - v1).normalize()
            for idx in (p1, p2, p3):
                if idx in pt_per_normal:
                    n_ = pt_per_normal[idx]
                    n_ = (n_ + normal) * 0.5 if Vector3.dot(normal, n_) > 0.0 else (-n_ + normal) * 0.5
                    pt_per_normal[idx] = n_.normalize()
                else:
                    pt_per_normal.update({idx: normal})
        self.normals = tuple(pt_per_normal[idx] for idx in range(len(self.vertices)))

    def draw(self, axis=None, show: bool = True):
        axis = axis if axis else plt.axes(projection='3d')
        faces = self.faces_array
        vertices = self.vertices_array
        axis.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, shade=True)
        axis.set_xlabel("x, [mm]")
        axis.set_ylabel("y, [mm]")
        axis.set_zlabel("z, [mm]")
        if show:
            axis.set_aspect('equal', 'box')
            plt.show()
        return axis

    def draw_wire_frame(self, axis=None, color="k", show: bool = True):
        axis = axis if axis else plt.axes(projection='3d')
        if self.triangulated:
            for face in self.faces:
                (i1, _, _), (i2, _, _), (i3, _, _) = face
                p1, p2, p3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
                axis.plot((p1.x, p2.x, p3.x, p1.x), (p1.y, p2.y, p3.y, p1.y), (p1.z, p2.z, p3.z, p1.z), color)
        else:
            for face in self.faces:
                (i1, _, _), (i2, _, _), (i3, _, _), (i4, _, _) = face
                p1, p2, p3, p4 = self.vertices[i1], self.vertices[i2], self.vertices[i3], self.vertices[i4]
                axis.plot((p1.x, p2.x, p3.x, p4.x, p1.x),
                          (p1.y, p2.y, p3.y, p4.y, p1.y),
                          (p1.z, p2.z, p3.z, p4.z, p1.z), color)
        if show:
            axis.set_aspect('equal', 'box')
            plt.show()
        return axis

    def align_2_center(self) -> None:
        center = self.bounds.center * Vector3(1, 0, 1)
        if (center - Vector3(0, 0, 0)).magnitude <  1e-6:
            return  # centered

        for v in self.vertices:
            v -= center

        _min = self.bounds.min - center
        _max = self.bounds.max - center
        self.bounds = BoundingBox()
        self.bounds.encapsulate(_min)
        self.bounds.encapsulate(_max)

    @staticmethod
    def _calc_indices(index: int, stride: int, shift: int) -> Tuple[int, ...]:
        row, col = divmod(index, stride - 1)
        p1 = shift + 0 + col + row * stride
        p2 = shift + 1 + col + row * stride
        p3 = shift + 1 + col + (row + 1) * stride
        p4 = shift + 0 + col + (row + 1) * stride
        return p1, p2, p3, p4

    @classmethod
    def _build_mesh_from_shape(cls, shape: ParametricSurface, indices_shift: int = 0,
                               triangulate: bool = False) -> 'Mesh':
        rows, cols = shape.resolution
        n_points = rows * cols
        uvs = []
        for index in range(n_points):
            row, col = divmod(index, cols)
            uvs.append(Vector2(float(row) / (rows - 1), float(col) / (cols - 1)))
        positions = [shape.point(uv) for uv in uvs]
        normals = [shape.normal(uv) for uv in uvs]
        faces = []
        if triangulate:
            if shape.surface_orientation() > 0.0:
                for index in range((rows - 1) * (cols - 1)):
                    p1, p2, p3, p4 = Mesh._calc_indices(index, cols, indices_shift)
                    faces.append(((p1, p1, p1), (p3, p3, p3), (p2, p2, p2)))
                    faces.append(((p1, p1, p1), (p4, p4, p4), (p3, p3, p3)))
            else:
                for index in range((rows - 1) * (cols - 1)):
                    p1, p2, p3, p4 = Mesh._calc_indices(index, cols, indices_shift)
                    faces.append(((p1, p1, p1), (p2, p2, p2), (p3, p3, p3)))
                    faces.append(((p1, p1, p1), (p3, p3, p3), (p4, p4, p4)))
        else:
            if shape.surface_orientation() > 0.0:
                for index in range((rows - 1) * (cols - 1)):
                    p1, p2, p3, p4 = Mesh._calc_indices(index, cols, indices_shift)
                    faces.append(((p1, p1, p1), (p4, p4, p4), (p3, p3, p3), (p2, p2, p2)))
            else:
                for index in range((rows - 1) * (cols - 1)):
                    p1, p2, p3, p4 = Mesh._calc_indices(index, cols, indices_shift)
                    faces.append(((p1, p1, p1), (p2, p2, p2), (p3, p3, p3), (p4, p4, p4)))
        # return positions, normals, uvs, faces
        mesh = cls()
        mesh.triangulated = triangulate
        mesh.vertices = tuple(positions)
        mesh.normals = tuple(normals)
        mesh.uvs = tuple(uvs)
        mesh.faces = tuple(faces)
        return mesh

    @classmethod
    def _make_cup(cls, points: Union[List[Vector3], Tuple[Vector3, ...]]) -> 'Mesh':
        normal = sum(Vector3.cross(b - a, c - b) for a, b, c in zip(points[:-2], points[1:-1], points[2:])).normalize()
        center = sum(p for p in points) / len(points)
        right = (points[0] - center).normalize()
        front = Vector3.cross(normal, right).normalize()
        transform = Matrix4.build_transform(right, normal, front, center)
        transform_inv = transform.inverted
        pts = tuple(transform_inv.multiply_by_point(p).xz for p in points)
        polygon = triangulate_polygon(pts)
        mesh = cls()
        mesh.triangulated = True
        mesh.vertices = tuple(transform.multiply_by_point(Vector3(x, 0.0, y)) for (x, y) in polygon.vertices)
        mesh.normals = tuple(normal for _ in polygon.vertices)
        mesh.uvs = polygon.uvs
        mesh.faces = polygon.faces
        return mesh

    @classmethod
    def bevel_mesh(cls, resolution, start_shape, path_shape, end_shape=None,
                   params_mask: int = BEVEL_CUP) -> Tuple['Mesh', ...]:
        shape = BevelSurface(start_shape, path_shape, end_shape)
        shape.resolution = resolution
        mesh = Mesh._build_mesh_from_shape(shape, 0, (params_mask & MESH_TRIANGULATE) != 0)
        meshes = [mesh]
        dv = 1.0 / (shape.resolution[1] - 1)
        if params_mask & BEVEL_CUP_START:
            points = tuple(shape.point(Vector2(0.0, dv * index)) for index in reversed(range(shape.resolution[1] - 1)))
            meshes.append(Mesh._make_cup(points))
        if params_mask & BEVEL_CUP_END:
            points = tuple(shape.point(Vector2(1.0, dv * index)) for index in range(shape.resolution[1] - 1))
            meshes.append(Mesh._make_cup(points))
        return tuple(meshes)

    @classmethod
    def lathe_mesh(cls, resolution, profile, start_angle: float = .0, end_angle: float = TWO_PI, shift: float = .0,
                   params_mask: int = LATHE_CUP) -> Tuple['Mesh', ...]:
        shape = LatheSurface(profile)
        shape.resolution = resolution
        shape.axis_offset = shift
        shape.lathe_angle = end_angle - start_angle
        shape.start_angle = start_angle
        meshes = [Mesh._build_mesh_from_shape(shape, 0, (params_mask & MESH_TRIANGULATE) != 0)]
        dv = 1.0 / (shape.resolution[0] - 1)
        if params_mask & LATHE_CUP_START:
            points = tuple(shape.point(Vector2(dv * index, 0.0)) for index in reversed(range(shape.resolution[0] )))
            meshes.append(Mesh._make_cup(points))
        if params_mask & LATHE_CUP_END:
            points = tuple(shape.point(Vector2(dv * index, 1.0)) for index in range(shape.resolution[0]))
            meshes.append(Mesh._make_cup(points))
        if params_mask & LATHE_CUP_INTERNAL:
            cyl_shape = CylinderSurface()
            cyl_shape.start_angle = shape.start_angle
            cyl_shape.angle = shape.lathe_angle
            cyl_shape.radius2 = cyl_shape.radius1 = shape.axis_offset
            cyl_shape.height = profile[-1].z - profile[0].z
            cyl_shape.resolution = (resolution[-1], 3,)
            cyl = Mesh._build_mesh_from_shape(cyl_shape, 0,  (params_mask & MESH_TRIANGULATE) != 0)
            meshes.append(cyl)
        return tuple(meshes)
