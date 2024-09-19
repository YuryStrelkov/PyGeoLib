from ...Transformations.transform_3d import Transform3d
from ...Matrices.matrix4 import Matrix4
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from typing import List, Union, Tuple


def bezier_curve_3d(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3, t) -> Vector3:
    one_min_t: float = 1.0 - t
    a: float = one_min_t * one_min_t * one_min_t
    b: float = 3.0 * one_min_t * one_min_t * t
    c: float = 3.0 * one_min_t * t * t
    d: float = t * t * t
    return Vector3(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                   p1.y * a + p2.y * b + p3.y * c + p4.y * d,
                   p1.z * a + p2.z * b + p3.z * c + p4.z * d)


def bezier_tangent_3d(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3, t) -> Vector3:
    d: float = 3.0 * t * t
    a: float = -3.0 + 6.0 * t - d
    b: float = 3.0 - 12.0 * t + 3.0 * d
    c: float = 6.0 * t - 3.0 * d
    return Vector3(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                   p1.y * a + p2.y * b + p3.y * c + p4.y * d,
                   p1.z * a + p2.z * b + p3.z * c + p4.z * d)


def bezier_curve_2d(p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2, t) -> Vector2:
    one_min_t: float = 1.0 - t
    a: float = one_min_t * one_min_t * one_min_t
    b: float = 3.0 * one_min_t * one_min_t * t
    c: float = 3.0 * one_min_t * t * t
    d: float = t * t * t
    return Vector2(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                   p1.y * a + p2.y * b + p3.y * c + p4.y * d)


def bezier_tangent_2d(p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2, t) -> Vector2:
    d: float = 3.0 * t * t
    a: float = -3.0 + 6.0 * t - d
    b: float = 3.0 - 12.0 * t + 3.0 * d
    c: float = 6.0 * t - 3.0 * d
    return Vector2(p1.x * a + p2.x * b + p3.x * c + p4.x * d,
                   p1.y * a + p2.y * b + p3.y * c + p4.y * d)


def bezier_interpolate_pt_3d(param_t: float, points: Union[List[Vector3], Tuple[Vector3, ...]]) -> Vector3:
    t = param_t * ((len(points)) // 3)
    scet_index = int(t - 1e-6)
    t = t - int(t - 1e-6)
    n_points = len(points) - 1
    p1 = points[min(scet_index * 3, n_points)]
    p2 = points[min(scet_index * 3 + 1, n_points)]
    p3 = points[min(scet_index * 3 + 2, n_points)]
    p4 = points[min(scet_index * 3 + 3, n_points)]
    return bezier_curve_3d(p1, p2, p3, p4, t)


def bezier_interpolate_tangent_3d(param_t: float, points: Union[List[Vector3], Tuple[Vector3, ...]]) -> Vector3:
    t = param_t * ((len(points)) // 3)
    scet_index = int(t - 1e-6)
    t = t - int(t - 1e-6)
    n_points = len(points) - 1
    p1 = points[min(scet_index * 3, n_points)]
    p2 = points[min(scet_index * 3 + 1, n_points)]
    p3 = points[min(scet_index * 3 + 2, n_points)]
    p4 = points[min(scet_index * 3 + 3, n_points)]
    return bezier_tangent_3d(p1, p2, p3, p4, t)


def bezier_interpolate_pt_2d(param_t: float, points: Union[List[Vector2], Tuple[Vector2, ...]]) -> Vector2:
    t = param_t * ((len(points)) // 3)
    scet_index = int(t - 1e-6)
    t = t - int(t - 1e-6)
    p1 = points[scet_index * 3]
    p2 = points[scet_index * 3 + 1]
    p3 = points[scet_index * 3 + 2]
    p4 = points[scet_index * 3 + 3]
    return bezier_curve_2d(p1, p2, p3, p4, t)


BEZIER_ANCHOR_VAL = 0.552284749831

BEZIER_HELICS_ANCHOR_VAL = 0.552284749831 * 0.707

BEZIER_CIRCLE_POINTS = (Vector3(1.0, 0.0, 0.0),
                        Vector3(1.0, BEZIER_ANCHOR_VAL, 0.0),
                        Vector3(BEZIER_ANCHOR_VAL, 1.0, 0.0),
                        Vector3(0.0, 1.0, 0.0),  # 2 sect
                        Vector3(-BEZIER_ANCHOR_VAL, 1.0, 0.0),
                        Vector3(-1.0, BEZIER_ANCHOR_VAL, 0.0),
                        Vector3(-1.0, 0.0, 0.0),  # 3 sect
                        Vector3(-1.0, -BEZIER_ANCHOR_VAL, 0.0),
                        Vector3(-BEZIER_ANCHOR_VAL, -1.0, 0.0),
                        Vector3(0.0, -1.0, 0.0),
                        Vector3(BEZIER_ANCHOR_VAL, -1.0, 0.0),
                        Vector3(1.0, -BEZIER_ANCHOR_VAL, 0.0),
                        Vector3(1.0, 0.0, 0.0),)


BEZIER_HELICS_POINTS = (Vector3(0.0, 1.0, 0.0),
                        Vector3(0.086806, 1.0, 0.54542),
                        Vector3(0.16319, 0.54542, 1.0),
                        Vector3(0.25, 0.0, 1.0),
                        Vector3(0.33681, -0.54542, 1.0),
                        Vector3(0.41319, -1.0, 0.54542),
                        Vector3(0.5, -1.0, 0.0),
                        Vector3(0.58681, -1.0, -0.54542),
                        Vector3(0.66319, -0.54542, -1.0),
                        Vector3(0.75, 0.0, -1.0),
                        Vector3(0.83681, 0.54542, -1.0),
                        Vector3(0.91319, 1.0, -0.54542),
                        Vector3(1.0, 1.0, 0.0),)


def bezier_circle_3d(position: Vector3 = None, radius: float = 1.0, orientation: float = 1):
    transform = Transform3d()
    if position:
        transform.origin = position
    transform.scale = Vector3(radius, radius, radius)
    if orientation == 1:
        transform.angles = Vector3(90.0, 0.0, 0.0)
    if orientation == 2:
        transform.angles = Vector3(0.0, 0.0, 90.0)

    return tuple(transform.transform_vect(v, 1.0) for v in BEZIER_CIRCLE_POINTS) if position \
        else tuple(transform.transform_vect(v, 1.0) for v in BEZIER_CIRCLE_POINTS)


def bezier_helics_3d(position: Vector3 = None, radius: float = 1.0,
                     gap: float = 1, turns: float = 3.5, orientation: float = 0):
    position = position if position else Vector3(0.0, 0.0, 0.0)
    if orientation == 0:
        transform = Matrix4(gap / turns, 0.0, 0.0, position.x,
                            0.0, radius, 0.0, position.y,
                            0.0, 0.0, radius, position.z,
                            0.0, 0.0, 0.0, 1.0)
        up_dir = Vector3(gap * turns, 0.0, 0.0)
        vert_scale = Vector3(gap * turns, 1.0, 1.0)
    elif orientation == 1:
        transform = Matrix4(0.0, 0.0, radius, position.x,
                            gap / turns, 0.0, 0.0, position.y,
                            0.0, radius, 0.0, position.z,
                            0.0, 0.0, 0.0, 1.0)
        up_dir = Vector3(gap * turns, 0.0, 0.0)
        vert_scale = Vector3(gap * turns, 1.0, 1.0)
    elif orientation == 2:
        transform = Matrix4(0.0, 0.0, radius, position.x,
                            0.0, radius, 0.0, position.y,
                            gap / turns, 0.0, 0.0, position.z,
                            0.0, 0.0, 0.0, 1.0)
        up_dir = Vector3(gap * turns, 0.0, 0.0)
        vert_scale = Vector3(gap * turns, 1.0, 1.0)
    else:
        transform = Matrix4(gap / turns, 0.0, 0.0, position.x,
                            0.0, radius, 0.0, position.y,
                            0.0, 0.0, radius, position.z,
                            0.0, 0.0, 0.0, 1.0)
        up_dir = Vector3(gap * turns, 0.0, 0.0)
        vert_scale = Vector3(gap * turns, 1.0, 1.0)

    points = [transform.multiply_by_point(BEZIER_HELICS_POINTS[0])]
    for index in range(int(turns)):
        shift = up_dir * index
        for v in BEZIER_HELICS_POINTS[1:]:
            points.append(transform.multiply_by_point(v * vert_scale + shift))
    dt = turns - int(turns)
    if dt > 1e-5:
        shift = up_dir * int(turns)
        o1 = shift + bezier_interpolate_pt_3d(dt * 0.0, BEZIER_HELICS_POINTS) * vert_scale
        o2 = shift + bezier_interpolate_pt_3d(dt * 0.25, BEZIER_HELICS_POINTS) * vert_scale
        o3 = shift + bezier_interpolate_pt_3d(dt * 0.5, BEZIER_HELICS_POINTS) * vert_scale
        o4 = shift + bezier_interpolate_pt_3d(dt * 0.75, BEZIER_HELICS_POINTS) * vert_scale
        o5 = shift + bezier_interpolate_pt_3d(dt * 1.0, BEZIER_HELICS_POINTS) * vert_scale

        vert_scale *= 0.5 * dt * BEZIER_ANCHOR_VAL
        t1 = bezier_interpolate_tangent_3d(dt * 0.0, BEZIER_HELICS_POINTS) * vert_scale
        t2 = bezier_interpolate_tangent_3d(dt * 0.25, BEZIER_HELICS_POINTS) * vert_scale
        t3 = bezier_interpolate_tangent_3d(dt * 0.5, BEZIER_HELICS_POINTS) * vert_scale
        t4 = bezier_interpolate_tangent_3d(dt * 0.75, BEZIER_HELICS_POINTS) * vert_scale
        t5 = bezier_interpolate_tangent_3d(dt * 1.0, BEZIER_HELICS_POINTS) * vert_scale

        points.append(transform.multiply_by_point(o1 + t1))
        points.append(transform.multiply_by_point(o2 - t2))
        points.append(transform.multiply_by_point(o2))
        points.append(transform.multiply_by_point(o2 + t2))
        points.append(transform.multiply_by_point(o3 - t3))
        points.append(transform.multiply_by_point(o3))
        points.append(transform.multiply_by_point(o3 + t3))
        points.append(transform.multiply_by_point(o4 - t4))
        points.append(transform.multiply_by_point(o4))
        points.append(transform.multiply_by_point(o4 + t4))
        points.append(transform.multiply_by_point(o5 - t5))
        points.append(transform.multiply_by_point(o5))
    return points

