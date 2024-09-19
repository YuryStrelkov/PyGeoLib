from ..common import TWO_PI
from ..Surfaces.Curves import triangulate_polygon
from ..Surfaces import BevelSurface, ParametricSurface, LatheSurface, CylinderSurface
from ..Vectors import Vector3, Vector2
from matplotlib import pyplot as plt
from typing import Union, Tuple, List
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


class Mesh:
    __slots__ = ('vertices', 'uvs', 'faces', 'normals', 'triangulated')

    def __init__(self):
        self.triangulated = True
        self.vertices: Union[Tuple[Vector3], None] = None
        self.faces: Union[Tuple[Tuple[int, ...]], None] = None
        self.uvs: Union[Tuple[Vector3], None] = None
        self.normals: Union[Tuple[Vector3], None] = None

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

    def draw(self, axis=None, color="k"):
        axis = axis if axis else plt.axes(projection='3d')
        if self.triangulated:
            for face in self.faces:
                i1, i2, i3 = face
                p1, p2, p3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
                axis.plot((p1.x, p2.x, p3.x, p1.x), (p1.y, p2.y, p3.y, p1.y), (p1.z, p2.z, p3.z, p1.z), color)
        else:
            for face in self.faces:
                i1, i2, i3, i4 = face
                p1, p2, p3, p4 = self.vertices[i1], self.vertices[i2], self.vertices[i3], self.vertices[i4]
                axis.plot((p1.x, p2.x, p3.x, p4.x, p1.x),
                          (p1.y, p2.y, p3.y, p4.y, p1.y),
                          (p1.z, p2.z, p3.z, p4.z, p1.z), color)
        return axis

    @staticmethod
    def _calc_indices(index: int, stride: int, shift: int) -> Tuple[int, ...]:
        row, col = divmod(index, stride - 1)
        p1 = shift + 0 + col + row * stride
        p2 = shift + 1 + col + row * stride
        p3 = shift + 1 + col + (row + 1) * stride
        p4 = shift + 0 + col + (row + 1) * stride
        return p1, p2, p3, p4

    @classmethod
    def _build_mesh_from_shape(cls, shape: ParametricSurface, indices_shift: int = 0, triangulate: bool = False) -> 'Mesh':
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
                    faces.append((p1, p3, p2))
                    faces.append((p1, p4, p3))
            else:
                for index in range((rows - 1) * (cols - 1)):
                    p1, p2, p3, p4 = Mesh._calc_indices(index, cols, indices_shift)
                    faces.append((p1, p2, p3))
                    faces.append((p1, p3, p4))
        else:
            if shape.surface_orientation() > 0.0:
                for index in range((rows - 1) * (cols - 1)):
                    p1, p2, p3, p4 = Mesh._calc_indices(index, cols, indices_shift)
                    faces.append((p1, p4, p3, p2))
            else:
                for index in range((rows - 1) * (cols - 1)):
                    p1, p2, p3, p4 = Mesh._calc_indices(index, cols, indices_shift)
                    faces.append((p1, p2, p3, p4))
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
