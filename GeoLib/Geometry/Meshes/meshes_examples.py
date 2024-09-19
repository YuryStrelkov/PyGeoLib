from ..Meshes.obj_file_builder import build_meshes_obj_file, build_polygon_obj_file
from ..Surfaces.Curves import squares_marching_2d, triangulate_polygons
from ..Meshes.cubes_marching import cubes_marching
from ..Meshes.mesh import Mesh
from matplotlib import pyplot as plt
from ..Surfaces.Curves.bezier import BEZIER_CIRCLE_POINTS
from ..Vectors import Vector2, Vector3
from math import cos, sqrt
from ..color import Color
from ..common import PI
import numpy as np


def cubes_function(x: float, y: float, z: float) -> float:
    value =  sqrt(x * x * 1.0 + y * y * 1.0 + z * z * 1.0)
    value = min(sqrt((x - 1) * (x - 1) + y * y + z * z), value)
    value = min(sqrt(x * x + (y - 1) * (y - 1) + z * z), value)
    value = min(sqrt(x * x + y * y + (z - 1) * (z - 1)), value)
    value = min(sqrt(x * x + y * y + (z - 2) * (z - 2)), value)
    value = min(sqrt(x * x + y * y + (z - 3) * (z - 3)), value)
    value = min(sqrt(x * x + y * y + (z - 4) * (z - 4)), value)
    # value = min(sqrt(x * x + y * y + (z - 5) * (z - 5)), value)
    # value = min(sqrt(x * x + y * y + (z - 6) * (z - 6)), value)
    return value


def ear_clipping_example():
    scale = 100
    t = np.linspace(0.0, PI, 128)
    field = np.array(tuple(tuple(cos(sqrt(scale * x * x * x + scale * (y + 1.0) * y ** 2)) for y in t) for x in t))
    sections = squares_marching_2d(field, Vector2(0.0, 0.0), Vector2(t[-1], t[-1]),
                                   (t.size, t.size), threshold=0.5, interpolate=True)
    polygons = triangulate_polygons(sections)
    with open("polygon shapes.obj", "wt") as shape:
        build_polygon_obj_file(sections, shape)

    axes = plt.gca()
    cmap = tuple(c.matplotlib_color_code for c in Color.color_map_quadratic(len(sections)))
    for index, (points, polygon) in enumerate(zip(sections, polygons)):
        axes = polygons[index].draw(axes, cmap[index])
        axes.plot(tuple(p.x for p in points), tuple(p.y for p in points))
    axes.axis('equal')
    plt.show()


def cube_marching_example():
    mesh = cubes_marching(cubes_function, threshold=1.0)
    with open("cubes marching mesh.obj", "wt") as shape:
        build_meshes_obj_file((mesh,), shape)
    mesh.draw()
    plt.gca().axis('equal')
    plt.show()


def shape_based_meshes_example():
    meshes = Mesh.bevel_mesh((32, 16), tuple(0.333 * p for p in BEZIER_CIRCLE_POINTS),
                             tuple(Vector3(p.x, 0.0, p.y) for p in BEZIER_CIRCLE_POINTS[:7]))
    with open("bevel_mesh.obj", "wt") as shape:
        build_meshes_obj_file(meshes, shape)
    axis = None
    for mesh in meshes:
        axis = mesh.draw(axis)
    plt.gca().axis('equal')
    plt.show()


def lathe_shape_meshes_example():
    meshes = Mesh.lathe_mesh((32, 16), (Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 1), Vector3(0, 0, 1),), 3.1415, 1.0)
    with open("lathe_mesh.obj", "wt") as shape:
        build_meshes_obj_file(meshes, shape)
    axis = None
    for mesh in meshes:
        axis = mesh.draw(axis)
    plt.gca().axis('equal')
    plt.show()