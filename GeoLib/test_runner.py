from typing import Iterable, Union, List, Tuple
from matplotlib import pyplot as plt
import numpy as np

from Geometry import bevel_shape_example, bezier_path_example, torus_example, \
    helix_example, lathe_shape_example, shape_to_obj_file_example, squares_marching_example, \
    ear_clipping_example, cube_marching_example, surfaces_tracing_test, tracing_2d_test, tracing_3d_test, \
    shape_based_meshes_example, lathe_shape_meshes_example, Vector3, bezier_interpolate_pt_3d, Mesh, PI, \
    build_meshes_obj_file


def convert_to_bezier(points: Union[List[Vector3], Tuple[Vector3, ...]], anchor: float = 0.25):
    bezier_points = []
    for p1, p2, p3 in zip(points[:-2], points[1:-1], points[2:]):
        dp1 = (p2 - p1).normalize()
        dp2 = (p3 - p2).normalize()
        bezier_points.append(p2 - dp1 * anchor)
        bezier_points.append(p2)
        bezier_points.append(p2 + dp2 * anchor)
    bezier_points.append((points[-1] + points[-2]) * 0.5)
    bezier_points.append((points[-1] + points[-2]) * 0.5)
    bezier_points.append(points[-1])
    bezier_points.insert(0, (points[0] + points[1]) * 0.5)
    bezier_points.insert(0, (points[0] + points[1]) * 0.5)
    bezier_points.insert(0, points[0])
    return tuple(bezier_points)


from UI.ui_example import UIExample


if __name__ == '__main__':
    # cube_marching_example()
    UIExample.create_and_run()
    a = np.array(tuple(f for f in ((1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3))), dtype=int)

    # shape_to_obj_file_example()
    exit(1)
    radius = 3.0

    points0 = [Vector3(0.0, radius,        0.0),
               Vector3(0.0, radius,      - 2.0),
               Vector3(0.0, radius - 2,  - 2.0),
               Vector3(0.0, radius - 2,   -3.0),
               Vector3(0.0, 0.0,          -3.0)]

    points1 = [Vector3(0.0, radius + 0.5 * 0.0, 0.0),
               Vector3(0.0, radius + 0.5 * 1.0, 0.0),
               Vector3(0.0, radius + 0.5 * 1.0, 1.0),
               Vector3(0.0, radius + 0.5 * 1.0, 4.0),
               Vector3(0.0, radius + 0.5 * 1.0, 5.0),
               Vector3(0.0, radius + 0.5 * 2.0, 5.0),
               Vector3(0.0, radius + 0.5 * 2.0, 6.0),
               Vector3(0.0, radius + 0.5 * 0.0, 6.0)]

    points2 = [Vector3(0.0, radius, 0.0),
               Vector3(0.0, radius, 2.5),
               Vector3(0.0, radius, 2.5),
               Vector3(0.0, radius, 6.0)]

    points3 = [Vector3(0.0, radius + 0.0, 6.0),
               Vector3(0.0, radius + 0.0, 8.0),
               Vector3(0.0, radius + 1.5, 9.5),
               Vector3(0.0, radius + 1.0, 10.0),
               Vector3(0.0, radius + -0.7, 8.5),
               Vector3(0.0, radius + -0.7, 0.5),
               Vector3(0.0, radius + -2.0, -1.0),
               Vector3(0.0, radius + -3.0, -1.0)]

    b_points0 = convert_to_bezier(points0)
    b_points1 = convert_to_bezier(points1)
    b_points2 = convert_to_bezier(points2)
    b_points3 = convert_to_bezier(points3)

    shape1 = Mesh.lathe_mesh((32, 17), b_points1, -PI * 0.25, PI * 0.25, shift=2.0, params_mask=3)
    shape2 = Mesh.lathe_mesh((32, 17), b_points1, PI - PI * 0.25, PI + PI * 0.25, shift=2.0, params_mask=3)
    shape3 = Mesh.lathe_mesh((32, 17 * 4), b_points0, shift=-2.0, params_mask=4)
    shape4 = Mesh.lathe_mesh((3, 17), points2, PI * 0.5 - PI * 0.25, PI * 0.5 + PI * 0.25, shift=2.0, params_mask=0)
    shape5 = Mesh.lathe_mesh((3, 17), points2, -PI * 0.5 - PI * 0.25, -PI * 0.5 + PI * 0.25, shift=2.0, params_mask=0)
    shape6 = Mesh.lathe_mesh((128, 17 * 4), b_points3, 0.0, 2*PI, shift=-2.0, params_mask=4)
    meshes = (*shape1, *shape2, *shape3, *shape4, *shape5, *shape6)

    with open("lamp_mesh.obj", "wt") as shape:
        build_meshes_obj_file(meshes, shape)

    axis = None
    for mesh in meshes:
        axis = mesh.draw(axis)
    plt.gca().set_aspect('equal', 'box')
    plt.show()

    t = np.linspace(0.0, 1.0, 128)
    for points, b_points in zip((points0, points1, points3), (b_points0, b_points1, b_points3)):
        pts = [bezier_interpolate_pt_3d(ti, b_points) for ti in t.flat]
        plt.plot(tuple(v.y for v in points), tuple(v.z for v in points), "k:o")
        plt.plot(tuple(v.y for v in b_points), tuple(v.z for v in b_points), "r:*")
        plt.plot(tuple(v.y for v in pts), tuple(v.z for v in pts), "b")
    plt.gca().set_aspect('equal', 'box')
    plt.show()
    # shape_based_meshes_example()
    # lathe_shape_meshes_example()
    # tracing_3d_test()
    # surfaces_tracing_test()
    # tracing_2d_test()
    # bevel_shape_example()
    # bezier_path_example()
    # torus_example()
    # helix_example()
    # lathe_shape_example()
    # shape_to_obj_file_example()
    # squares_marching_example()
    # ear_clipping_example()
    # cube_marching_example()
