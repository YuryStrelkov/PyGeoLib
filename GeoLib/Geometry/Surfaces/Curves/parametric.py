from ...Matrices.matrix4 import Matrix4
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from ...common import TWO_PI
from typing import Tuple
from math import sin, cos


def ellipse_x(param_t: float, pos: Vector3, axis: Vector2) -> Vector3:
    return Vector3(pos.x, pos.y + axis.x * cos(param_t * TWO_PI), pos.z + axis.y * sin(param_t * TWO_PI))


def ellipse_y(param_t: float, pos: Vector3, axis: Vector2) -> Vector3:
    return Vector3(pos.x + axis.x * cos(param_t * TWO_PI), pos.y, pos.z + axis.y * sin(param_t * TWO_PI))


def ellipse_z(param_t: float, pos: Vector3, axis: Vector2) -> Vector3:
    return Vector3(pos.x + axis.x * cos(param_t * TWO_PI), pos.y + axis.y * sin(param_t * TWO_PI), pos.z)


def create_ellipse(pos: Vector3 = None, axis: Vector2 = None, n_points: int = 32, orientation: int = 0):
    n_points = max(5, n_points)
    pos = pos if pos else Vector3(0, 0, 0)
    axis = axis if axis else Vector2(1, 2)
    da = 1.0 / (n_points - 1)
    if orientation == 0:
        return tuple(ellipse_x(i * da, pos, axis) for i in range(n_points))
    if orientation == 1:
        return tuple(ellipse_y(i * da, pos, axis) for i in range(n_points))
    if orientation == 2:
        return tuple(ellipse_z(i * da, pos, axis) for i in range(n_points))
    return tuple(ellipse_x(i * da, pos, axis) for i in range(n_points))


def ellipsoidal_helix_x(param_t: float, pos: Vector3, axis: Vector2, turns: float = 5.0, gap: float = 2.0) -> Vector3:
    return Vector3(pos.x + param_t * gap * turns, pos.y + axis.x * cos(param_t * TWO_PI * turns),
                   pos.z + axis.y * sin(param_t * TWO_PI * turns))


def ellipsoidal_helix_y(param_t: float, pos: Vector3, axis: Vector2, turns: float = 5.0, gap: float = 2.0) -> Vector3:
    return Vector3(pos.x + axis.x * cos(param_t * TWO_PI * turns), pos.y + param_t * gap * turns,
                   pos.z + axis.y * sin(param_t * TWO_PI * turns))


def ellipsoidal_helix_z(param_t: float, pos: Vector3, axis: Vector2, turns: float = 5.0, gap: float = 2.0) -> Vector3:
    return Vector3(pos.x + axis.x * cos(param_t * TWO_PI * turns), pos.y + axis.y * sin(param_t * TWO_PI * turns),
                   pos.z + param_t * gap * turns)


def create_ellipsoidal_helix(pos: Vector3 = None, axis: Vector2 = None, turns: float = 5,
                             gap: float = 2.0, n_points: int = 128, orientation: int = 1):
    n_points = max(5, n_points)
    pos = pos if pos else Vector3(0, 0, 0)
    axis = axis if axis else Vector2(1, 2)
    da = 1.0 / (n_points - 1)
    if orientation == 0:
        return tuple(ellipsoidal_helix_x(i * da, pos, axis, turns, gap) for i in range(n_points))
    if orientation == 1:
        return tuple(ellipsoidal_helix_y(i * da, pos, axis, turns, gap) for i in range(n_points))
    if orientation == 2:
        return tuple(ellipsoidal_helix_z(i * da, pos, axis, turns, gap) for i in range(n_points))
    return tuple(ellipsoidal_helix_x(i * da, pos, axis, turns, gap) for i in range(n_points))


def create_helix(pos: Vector3 = None, radius: float = 1.0, turns: float = 5,
                 gap: float = 2.0, n_points: int = 128, orientation: int = 1):
    return create_ellipsoidal_helix(pos, Vector2(radius, radius), turns, gap, n_points, orientation)


def create_circle(pos: Vector3 = None, radius: float = 1.0, n_points: int = 128, orientation: int = 1):
    return create_ellipse(pos, Vector2(radius, radius),  n_points, orientation)


def ellipse_local_tbn_x(param_t: float, pos: Vector3, axis: Vector2) -> Matrix4:
    origin = ellipse_x(param_t, pos, axis)
    up    = Vector3(1.0, 0.0, 0.0)
    front = Vector3(0.0, -axis.x * sin(param_t * TWO_PI), axis.y * cos(param_t * TWO_PI)).normalize()
    right = Vector3.cross(front, up).normalize()
    return Matrix4.build_transform(front, up, right, origin)


def ellipse_local_tbn_y(param_t: float, pos: Vector3, axis: Vector2) -> Matrix4:
    origin = ellipse_y(param_t, pos, axis)
    up    = Vector3(0.0, 1.0, 0.0)
    front = Vector3(-axis.x * sin(param_t * TWO_PI), 0.0, axis.y * cos(param_t * TWO_PI)).normalize()
    right = Vector3.cross(front, up).normalize()
    return Matrix4.build_transform(front, up, right, origin)


def ellipse_local_tbn_z(param_t: float, pos: Vector3, axis: Vector2) -> Matrix4:
    origin = ellipse_z(param_t, pos, axis)
    up    = Vector3(0.0, 0.0, 1.0)
    front = Vector3(-axis.x * sin(param_t * TWO_PI), axis.y * cos(param_t * TWO_PI), 0.0).normalize()
    right = Vector3.cross(front, up).normalize()
    return Matrix4.build_transform(front, up, right, origin)


def circle_local_tbn_x(param_t: float, pos: Vector3, radius: float) -> Matrix4:
    return ellipse_local_tbn_x(param_t, pos, Vector2(radius, radius))


def circle_local_tbn_y(param_t: float, pos: Vector3, radius: float) -> Matrix4:
    return ellipse_local_tbn_y(param_t, pos, Vector2(radius, radius))


def circle_local_tbn_z(param_t: float, pos: Vector3, radius: float) -> Matrix4:
    return ellipse_local_tbn_z(param_t, pos, Vector2(radius, radius))


def create_ellipse_local_tbn_s(pos: Vector3 = None, axis: Vector2 = None,
                               n_points: int = 32, orientation: int = 0) -> Tuple[Matrix4, ...]:
    n_points = max(5, n_points)
    pos = pos if pos else Vector3(0, 0, 0)
    axis = axis if axis else Vector2(1, 2)
    da = 1.0 / (n_points - 1)
    if orientation == 0:
        return tuple(ellipse_local_tbn_x(i * da, pos, axis) for i in range(n_points))
    if orientation == 1:
        return tuple(ellipse_local_tbn_y(i * da, pos, axis) for i in range(n_points))
    if orientation == 2:
        return tuple(ellipse_local_tbn_z(i * da, pos, axis) for i in range(n_points))
    return tuple(ellipse_local_tbn_x(i * da, pos, axis) for i in range(n_points))


def create_circle_local_tbn_s(pos: Vector3 = None, radius: float = 1.0,
                              n_points: int = 32, orientation: int = 0) -> Tuple[Matrix4, ...]:
    return create_ellipse_local_tbn_s(pos, Vector2(radius, radius), n_points, orientation)


def ellipsoidal_helix_local_tbn_x(param_t: float, pos: Vector3, axis: Vector2,
                                  turns: float = 5.0, gap: float = 2.0) -> Matrix4:
    origin = ellipsoidal_helix_x(param_t, pos, axis, turns, gap)
    up    = Vector3(1.0, 0.0, 0.0)
    front = Vector3(gap * turns / TWO_PI,
                    -turns * axis.x * sin(param_t * TWO_PI * turns),
                    turns * axis.y * cos(param_t * TWO_PI * turns)).normalize()
    right = Vector3.cross(front, up).normalize()
    up = Vector3.cross(right, front).normalize()
    return Matrix4.build_transform(front, up, right, origin)


def ellipsoidal_helix_local_tbn_y(param_t: float, pos: Vector3, axis: Vector2,
                                  turns: float = 5.0, gap: float = 2.0) -> Matrix4:
    origin = ellipsoidal_helix_y(param_t, pos, axis, turns, gap)
    up    = Vector3(0.0, 1.0, 0.0)
    front = Vector3(-turns * axis.x * sin(param_t * TWO_PI * turns),
                    gap * turns / TWO_PI,
                    turns * axis.y * cos(param_t * TWO_PI * turns)).normalize()
    right = Vector3.cross(front, up).normalize()
    up = Vector3.cross(right, front).normalize()
    return Matrix4.build_transform(front, up, right, origin)


def ellipsoidal_helix_local_tbn_z(param_t: float, pos: Vector3, axis: Vector2,
                                  turns: float = 5.0, gap: float = 2.0) -> Matrix4:
    origin = ellipsoidal_helix_z(param_t, pos, axis, turns, gap)
    up    = Vector3(0.0, 0.0, 1.0)
    front = Vector3(-turns * axis.x * sin(param_t * TWO_PI * turns),
                    turns * axis.y * cos(param_t * TWO_PI * turns),
                    param_t * gap * turns / TWO_PI).normalize()
    right = Vector3.cross(front, up).normalize()
    up = Vector3.cross(right, front).normalize()
    return Matrix4.build_transform(front, up, right, origin)


def create_ellipsoidal_helix_local_tbn_s(pos: Vector3 = None, axis: Vector2 = None, turns: float = 5,
                                         gap: float = 2.0, n_points: int = 128, orientation: int = 1) -> Tuple[Matrix4, ...]:
    n_points = max(5, n_points)
    pos = pos if pos else Vector3(0, 0, 0)
    axis = axis if axis else Vector2(1, 2)
    da = 1.0 / (n_points - 1)
    if orientation == 0:
        return tuple(ellipsoidal_helix_local_tbn_x(i * da, pos, axis, turns, gap) for i in range(n_points))
    if orientation == 1:
        return tuple(ellipsoidal_helix_local_tbn_y(i * da, pos, axis, turns, gap) for i in range(n_points))
    if orientation == 2:
        return tuple(ellipsoidal_helix_local_tbn_z(i * da, pos, axis, turns, gap) for i in range(n_points))
    return tuple(ellipsoidal_helix_local_tbn_x(i * da, pos, axis, turns, gap) for i in range(n_points))


def create_helix_local_tbn_s(pos: Vector3 = None, radius: float = 1.0, turns: float = 5,
                             gap: float = 2.0, n_points: int = 128, orientation: int = 1) -> Tuple[Matrix4, ...]:
    return create_ellipsoidal_helix_local_tbn_s(pos, Vector2(radius, radius), turns, gap, n_points, orientation)
