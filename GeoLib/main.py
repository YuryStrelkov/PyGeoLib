import time

from matplotlib import pyplot as plt

from Geometry import Transform2d, Vector2, TWO_PI
from typing import Tuple, Iterable
from math import sin, cos

VIEW_POINTS = (Vector2(0, 0), Vector2(0, 1), Vector2(1, 1), Vector2(1, 0), Vector2(0, 0))


def transform_points(transform: Transform2d, points: Iterable[Vector2]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    transformed_points = tuple(transform.transform_vect(p, 1.0) for p in points)
    xs = tuple(p.x for p in transformed_points)
    ys = tuple(p.y for p in transformed_points)
    return xs, ys


def inv_transform_points(transform: Transform2d, points: Iterable[Vector2]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    transformed_points = tuple(transform.inv_transform_vect(p, 1.0) for p in points)
    xs = tuple(p.x for p in transformed_points)
    ys = tuple(p.y for p in transformed_points)
    return xs, ys


def draw_transform(transform: Transform2d) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return transform_points(transform, VIEW_POINTS)


def draw_inv_transform(transform: Transform2d) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return inv_transform_points(transform, VIEW_POINTS)


def make_transforms_pair(position: Vector2, scale: Vector2, angle: float) -> Tuple[Transform2d, Transform2d]:
    transform_1 = Transform2d()
    transform_2 = Transform2d()
    transform_2.origin = position
    transform_2.scale = scale
    transform_2.az = angle
    return transform_1, transform_2


def graphic(n_points: int = 256) -> Tuple[Vector2, ...]:
    n_points = max(3, n_points)
    dt = TWO_PI / (n_points - 1.0)
    return tuple(Vector2(sin(dt * i),  cos(dt * i)) for i in range(n_points))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t1, t2 = make_transforms_pair(Vector2(0.5, 0.5), Vector2(.333, .333), 0.0)
    points = graphic()
    t1_points = transform_points(t1, points)
    # points_to_view_transform
    t2_points = inv_transform_points(t2, points)
    t1_rect = draw_transform(t1)
    t2_rect = draw_transform(t2)
    plt.plot(*t1_points, "r")
    plt.plot(*t2_points, "b")
    plt.plot(*t1_rect, ":k")
    plt.plot(*t2_rect, ":b")
    plt.gca().set_xlim((0.0, 1.0))
    plt.gca().set_ylim((0.0, 1.0))
    plt.gca().set_aspect('equal', 'box')
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
