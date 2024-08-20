from typing import Tuple, List, TextIO, Iterable, Union
from ..Shapes.BezierShapes.shape import Shape
from ..Shapes.Curves import triangulate_polygon
from ..Meshes.cubes_marching import Mesh
from ..Vectors.vector2 import Vector2
from ..Vectors.vector3 import Vector3


def calc_indices(index: int, stride: int, shift: int) -> Tuple[int, ...]:
    row, col = divmod(index, stride - 1)
    p1 = shift + 1 + col + row * stride
    p2 = shift + 2 + col + row * stride
    p3 = shift + 2 + col + (row + 1) * stride
    p4 = shift + 1 + col + (row + 1) * stride
    return p1, p2, p3, p4


def build_curve_obj_file(lines: Iterable[Iterable[Union[Vector3, Vector2]]], file: TextIO = None):
    indices_shift = 0
    for shape_id, shape in enumerate(lines):
        print("#", file=file)
        print(f"# object lines {shape_id}", file=file)
        print("#", file=file)
        vert_count = 0
        for pt in shape:
            vert_count += 1
            if isinstance(pt, Vector2):
                print(f"v  {pt.x:4.6} {pt.y:4.6} 0.0", file=file)
                continue
            if isinstance(pt, Vector3):
                print(f"v  {pt.x:4.6} {pt.y:4.6} {pt.z:4.6}", file=file)
                continue
        print(f"# {vert_count} vertices\n", file=file)
        print(f"g line{shape_id}", file=file)
        print(f"l {' '.join(str(1 + index + indices_shift) for index in range(vert_count))}\n", file=file)
        indices_shift += vert_count


def build_obj_file(shapes: Tuple[Shape, ...], file: TextIO = None, triangulate: bool = False):
    indices_shift = 0
    for shape_id, shape in enumerate(shapes):
        points_per_length, points_per_diam = shape.resolution
        n_points = points_per_length * points_per_diam
        print("#", file=file)
        print(f"# object shape{shape_id}", file=file)
        print("#", file=file)
        uvs = []
        for index in range(n_points):
            row, col = divmod(index, points_per_diam)
            uvs.append(Vector2(float(row) / (points_per_length - 1), float(col) / (points_per_diam - 1)))
        for uv in uvs:
            hp = shape.world_space_point(uv)
            print(f"v  {hp.x:4.6} {hp.y:4.6} {hp.z:4.6}", file=file)
        print(f"# {n_points} vertices", file=file)

        for uv in uvs:
            print(f"vt {uv.x:4.6} {uv.y:4.6}", file=file)
        print(f"# {n_points} texture coords", file=file)

        for uv in uvs:
            hn = shape.world_space_normal(uv)
            print(f"vn {hn.x:4.6} {hn.y:4.6} {hn.z:4.6}", file=file)
        print(f"# {n_points} vertex normals", file=file)

        print(f"g shape{shape_id}", file=file)
        if triangulate:
            if shape.surface_orientation() > 0.0:
                for index in range((points_per_length - 1) * (points_per_diam - 1)):
                    p1, p2, p3, p4 = calc_indices(index,  points_per_diam, indices_shift)
                    print(f"s {(index + 1) * 2}", file=file)
                    print(f"f {p1}/{p1}/{p1} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=file)
                    print(f"f {p1}/{p1}/{p1} {p4}/{p4}/{p4} {p3}/{p3}/{p3}", file=file)
            else:
                for index in range((points_per_length - 1) * (points_per_diam - 1)):
                    p1, p2, p3, p4 = calc_indices(index,  points_per_diam, indices_shift)
                    print(f"s {(index + 1) * 2}", file=file)
                    print(f"f {p1}/{p1}/{p1} {p2}/{p2}/{p2} {p3}/{p3}/{p3}", file=file)
                    print(f"f {p1}/{p1}/{p1} {p3}/{p3}/{p3} {p4}/{p4}/{p4}", file=file)
        else:
            print(f"s 1", file=file)
            if shape.surface_orientation() > 0.0:
                for index in range((points_per_length - 1) * (points_per_diam - 1)):
                    p1, p2, p3, p4 = calc_indices(index,  points_per_diam, indices_shift)
                    print(f"f {p1}/{p1}/{p1} {p4}/{p4}/{p4} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=file)
            else:
                for index in range((points_per_length - 1) * (points_per_diam - 1)):
                    p1, p2, p3, p4 = calc_indices(index,  points_per_diam, indices_shift)
                    print(f"f {p1}/{p1}/{p1} {p2}/{p2}/{p2} {p3}/{p3}/{p3} {p4}/{p4}/{p4}", file=file)
        print(f"# {(points_per_length - 1) * (points_per_diam - 1)} faces", file=file)
        indices_shift += n_points


def build_polygon_obj_file(polygons: Iterable[List[Vector2]], file: TextIO = None):
    indices_shift = 0
    for polygon_id, polygon_raw in enumerate(polygons):
        polygon = triangulate_polygon(polygon_raw)
        print("#", file=file)
        print(f"# object shape{polygon_id}", file=file)
        print("#", file=file)
        for v in polygon.vertices:
            print(f"v  {v.x:4.6} {v.y:4.6} { 0.0:4.6}", file=file)
        print(f"# {len(polygon.uvs)} vertices", file=file)

        for uv in polygon.uvs:
            print(f"vt {uv.x:4.6} {uv.y:4.6}", file=file)
        print(f"# {len(polygon.uvs)} texture coords", file=file)

        for _ in range(len(polygon.uvs)):
            print(f"vn {0.0:4.6} {0.0:4.6} {1.0:4.6}", file=file)
        print(f"# {len(polygon.uvs)} vertex normals", file=file)

        print(f"g shape{polygon_id}", file=file)
        for index, (p1, p2, p3) in enumerate(polygon.faces):
            p1 += indices_shift + 1
            p2 += indices_shift + 1
            p3 += indices_shift + 1
            print(f"f {p1}/{p1}/{p1} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=file)
        indices_shift += len(polygon.uvs)


def build_meshes_obj_file(meshes: Iterable[Mesh], file: TextIO = None):
    indices_shift = 0
    for mesh_id, mesh in enumerate(meshes):
        if any(v is None for v in (mesh.uvs, mesh.vertices, mesh.faces, mesh.normals)):
            continue
        print("#", file=file)
        print(f"# object shape{mesh_id}", file=file)
        print("#", file=file)
        for v in mesh.vertices:
            print(f"v  {v.x:4.6} {v.y:4.6} { v.z:4.6}", file=file)
        print(f"# {len(mesh.vertices)} vertices", file=file)

        for uv in mesh.uvs:
            print(f"vt {uv.x:4.6} {uv.y:4.6}", file=file)
        print(f"# {len(mesh.uvs)} texture coords", file=file)

        for n in mesh.normals:
            print(f"vn {n.x:4.6} {n.y:4.6} {n.z:4.6}", file=file)
        print(f"# {len(mesh.normals)} vertex normals", file=file)

        print(f"g shape{mesh_id}", file=file)
        if mesh.triangulated:
            for index, (p1, p2, p3) in enumerate(mesh.faces):
                p1 += indices_shift + 1
                p2 += indices_shift + 1
                p3 += indices_shift + 1
                print(f"f {p1}/{p1}/{p1} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=file)
        else:
            for index, (p1, p2, p3, p4) in enumerate(mesh.faces):
                p1 += indices_shift + 1
                p2 += indices_shift + 1
                p3 += indices_shift + 1
                p4 += indices_shift + 1
                print(f"f {p1}/{p1}/{p1} {p4}/{p4}/{p4} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=file)
        indices_shift += len(mesh.vertices)

