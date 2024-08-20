from ..Transformations import Transform3d
from ..Vectors import Vector2, Vector3
from typing import Tuple, Generator
from ..Matrices import Matrix4
from ..common import PI
import math


def rect_source(shape: Tuple[int, int] = None, size: Vector2 = None, transform: Transform3d = None) -> \
        Generator[Tuple[Vector3, Vector3], None, None]:
    rows, cols = shape if shape else 16, 16
    width, height = size if size else 25.0, 25.0
    dy, dx = width / (rows - 1), height / (cols - 1)
    transform_m = transform.transform_matrix if transform else Matrix4.identity()
    direction = Vector3(1.0, 0.0, 0.0)
    for index in range(rows * cols):
        row, col = divmod(index, cols)
        yield transform_m.multiply_by_direction(direction),\
              transform_m.multiply_by_point(Vector3(0.0, -0.5 * height + row * dy, -0.5 * width + col * dy))


def sphere_source(radius: float = 1.0, shape: Tuple[int, int] = None, uv0: Vector2 = None,
                  uv1: Vector2 = None, transform: Transform3d = None) -> Generator[Tuple[Vector3, Vector3], None, None]:
    rows, cols = shape if shape else 16, 16
    u0, v0 = uv0 if uv0 else -PI * 0.125, -PI * 0.125
    u1, v1 = uv1 if uv1 else PI * 0.125, PI * 0.125
    width = u1 - u0
    height = v1 - v0
    da, db = width / (rows - 1), height / (cols - 1)
    transform_m = transform.transform_matrix if transform else Matrix4.identity()
    for index in range(rows * cols):
        row, col = divmod(index, cols)
        a, b = 0.5 * height + row * da, 0.5 * width + col * db
        sin_a, sin_b = math.sin(a), math.sin(b)
        cos_a, cos_b = math.cos(a), math.cos(b)
        direction = Vector3(sin_a * cos_b, sin_a * sin_b, cos_a)
        yield transform_m.multiply_by_direction(direction),\
              transform_m.multiply_by_point(direction.normalize() * radius)
