import dataclasses
from typing import Tuple, List, TextIO, Iterable, Union, Dict
from ..Surfaces.Parametric.parametric_surface import ParametricSurface
from ..Surfaces.Curves import triangulate_polygon
from ..Meshes.cubes_marching import Mesh
from ..Vectors.vector2 import Vector2
from ..Vectors.vector3 import Vector3
from ..Bounds import BoundingBox


def calc_indices(index: int, stride: int, shift: int) -> Tuple[int, ...]:
    row, col = divmod(index, stride - 1)
    p1 = shift + 1 + col + row * stride
    p2 = shift + 2 + col + row * stride
    p3 = shift + 2 + col + (row + 1) * stride
    p4 = shift + 1 + col + (row + 1) * stride
    return p1, p2, p3, p4


def create_curve_obj_file(lines: Iterable[Iterable[Union[Vector3, Vector2]]], file: TextIO = None):
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


def create_obj_file(shapes: Tuple[ParametricSurface, ...], file: TextIO = None, triangulate: bool = False):
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


def create_polygon_obj_file(polygons: Iterable[List[Vector2]], file: TextIO = None):
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


def create_meshes_obj_file(meshes: Iterable[Mesh], file: TextIO = None):
    indices_shift = (1, 1, 1)
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
                p1 = tuple(p + shift for p, shift in zip(p1, indices_shift))
                p2 = tuple(p + shift for p, shift in zip(p2, indices_shift))
                p3 = tuple(p + shift for p, shift in zip(p3, indices_shift))
                print(f"f {'/'.join(str(p) for p in p1)} "
                      f"{'/'.join(str(p) for p in p2)} "
                      f"{'/'.join(str(p) for p in p3)}",
                      file=file)
        else:
            for index, (p1, p2, p3, p4) in enumerate(mesh.faces):
                p1 = tuple(p + shift for p, shift in zip(p1, indices_shift))
                p2 = tuple(p + shift for p, shift in zip(p2, indices_shift))
                p3 = tuple(p + shift for p, shift in zip(p3, indices_shift))
                p4 = tuple(p + shift for p, shift in zip(p4, indices_shift))
                print(f"f {'/'.join(str(p) for p in p1)} "
                      f"{'/'.join(str(p) for p in p4)} "
                      f"{'/'.join(str(p) for p in p3)} "
                      f"{'/'.join(str(p) for p in p2)}", file=file)
                # print(f"f {p1}/{p1}/{p1} {p4}/{p4}/{p4} {p3}/{p3}/{p3} {p2}/{p2}/{p2}", file=file)
        indices_shift = tuple(
            a + b for a, b in zip(indices_shift, (len(mesh.vertices), len(mesh.uvs), len(mesh.normals))))


def _parce_face(face_str: str, indices_shift: Tuple[int, ...]) -> \
        Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    f1, f2, f3 = face_str.split(' ')[-3:]
    f1 = tuple(int(v) - shift if len(v) != 0 else 0 for v, shift in zip(f1.split('/'), indices_shift))
    f2 = tuple(int(v) - shift if len(v) != 0 else 0 for v, shift in zip(f2.split('/'), indices_shift))
    f3 = tuple(int(v) - shift if len(v) != 0 else 0 for v, shift in zip(f3.split('/'), indices_shift))
    return f1, f2, f3


def read_obj_files(file_path: str) -> Dict[str, Mesh]:
    @dataclasses.dataclass
    class ObjMesh:
        def __init__(self):
            self.vertices = []
            self.normals = []
            self.faces = []
            self.uvs = []
            self.bounds = BoundingBox()

    mesh = None
    meshes = {}
    meshes_raw = {}
    indices_shift = (1, 1, 1)
    with open(file_path, 'rt') as input_file:
        for line in input_file:
            line = line.replace('\n', '')
            line = line.rstrip()
            line = line.lstrip()
            if 'object' in line:
                new_shift = (len(mesh.vertices), len(mesh.uvs), len(mesh.normals)) if mesh else (0, 0, 0)
                indices_shift = tuple(a + b for a, b in zip(indices_shift, new_shift))
                mesh = ObjMesh()
                meshes_raw.update({line.split(' ')[-1]: mesh})
                continue
            if line.startswith('v '):
                x, y, z = line.split(' ')[-3:]
                vertex = Vector3(x, y, z)
                mesh.vertices.append(vertex)
                mesh.bounds.encapsulate(vertex)
                continue
            if line.startswith('vt'):
                x, y = line.split(' ')[-2:]
                mesh.uvs.append(Vector2(x, y))
                continue
            if line.startswith('vn'):
                x, y, z = line.split(' ')[-3:]
                mesh.normals.append(Vector3(x, y, z))
                continue
            if line.startswith('f'):
                mesh.faces.append(_parce_face(line, indices_shift))
                continue
        for m_name, m_data in meshes_raw.items():
            m = Mesh()
            m.faces = tuple(m_data.faces)
            m.vertices = tuple(m_data.vertices)
            m.uvs = tuple(m_data.uvs)
            m.normals = tuple(m_data.normals)
            m.bounds = m_data.bounds
            if len(m_data.uvs) != len(m_data.vertices) and len(m_data.uvs) != 0:
                uvs = []
                for (_, uv1, _,), (_, uv2, _,), (_, uv3, _,) in m_data.faces:
                    uvs.append(m.uvs[uv1])
                    uvs.append(m.uvs[uv2])
                    uvs.append(m.uvs[uv3])
                m.uvs = tuple(uvs)

            if len(m_data.normals) != len(m_data.vertices) and len(m_data.normals) != 0:
                normals = []
                for (_, _, n1), (_, _, n2), (_, _, n3) in m_data.faces:
                    normals.append(m.normals[n1])
                    normals.append(m.normals[n2])
                    normals.append(m.normals[n3])
                m.normals = tuple(normals)
            meshes.update({m_name: m})
    return meshes


def write_obj_files(file_path: str, meshes: Iterable[Mesh]):
    with open(file_path, 'wt') as output_file:
        create_meshes_obj_file(meshes, output_file)


def write_stl(file_path: str, mesh: Mesh):
    with open(file_path, "wt") as output_stl:
        print(f"solid {mesh.name}", file=output_stl)
        for face in mesh.faces:
            normal = sum(mesh.normals[n_id] for (_, _,  n_id) in face).normalize()
            v1, v2, v3 = tuple(mesh.vertices[v_id]for (v_id, _, _) in face)
            print(f"  facet normal {normal.x:e} {normal.y:e} {normal.z:e}", file=output_stl)
            print(f"    outer loop", file=output_stl)
            print(f"      vertex {v1.x:e} {v1.y:e} {v1.z:e}", file=output_stl)
            print(f"      vertex {v2.x:e} {v2.y:e} {v2.z:e}", file=output_stl)
            print(f"      vertex {v3.x:e} {v3.y:e} {v3.z:e}", file=output_stl)
            print(f"    endloop", file=output_stl)
            print(f"  endfacet", file=output_stl)
        print(f"endsolid {mesh.name}", file=output_stl)


