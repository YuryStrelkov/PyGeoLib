from ..Transformations.transform_3d import Transform3d
from ..common import NUMERICAL_ACCURACY
from ..Vectors.vector3 import Vector3
from typing import Tuple
import numpy as np
import math

MATERIAL = 'material'
GLASS = 'glass'
MIRROR = 'mirror'
GLASS_PARAMS = 'glass-params'
SOURCE_OBJECT = 'object'
IMAGE_OBJECT = 'image'
DUMMY_OBJECT = 'dummy'


def _mesh_shape(size: float = 1.0, steps: int = 32):
    steps = max(3, steps)
    dt = size / (steps - 1)
    return [[Vector3(row * dt - size * 0.5, col * dt - size * 0.5, 0) for col in range(steps)]for row in range(steps)]


def build_cylinder_shape_3d(radius1, radius2, h_0,  d_h, transform: Transform3d = None, steps: int = 21):
    steps = max(5, steps)
    da = np.pi * 2 / (steps - 1)
    transform = transform if transform else Transform3d()
    top = [Vector3(radius1 * math.cos(da * idx), radius1 * math.sin(da * idx), h_0   ) for idx in range(steps)]
    low = [Vector3(radius2 * math.cos(da * idx), radius2 * math.sin(da * idx), h_0 + d_h) for idx in range(steps)]
    x_cords, y_cords, z_cords = [[], []], [[], []], [[], []]
    for t, l in zip(top, low):
        t = transform.transform_vect(t)
        l = transform.transform_vect(l)
        x_cords[0].append(t.x)
        x_cords[1].append(l.x)
        y_cords[0].append(t.y)
        y_cords[1].append(l.y)
        z_cords[0].append(t.z)
        z_cords[1].append(l.z)
    return np.array(x_cords), np.array(y_cords), np.array(z_cords)


def build_shape_3d(radius: float = 1.0, semi_diam: float = 1.0,
                   transform: Transform3d = None, steps: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(semi_diam) <= NUMERICAL_ACCURACY:
        return np.array([]), np.array([]), np.array([])
    points = _mesh_shape(min(semi_diam, 2 * abs(radius)), steps)
    transform = transform if transform else Transform3d()
    sgn = -1.0 if radius > 0 else 1.0
    radius = radius if abs(radius) > NUMERICAL_ACCURACY else 1.0 / NUMERICAL_ACCURACY
    x_cords = []
    y_cords = []
    z_cords = []
    for row in points:
        x_row = []
        y_row = []
        z_row = []
        for p in row:
            factor = (max(*abs(p))) * 2.0
            point = p.normalize() * factor
            point.z = -radius - sgn * math.sqrt((max(radius * radius - point.x * point.x - point.y * point.y, 0.0)))
            point = transform.transform_vect(point, 1.0)
            x_row.append(point.x)
            y_row.append(point.y)
            z_row.append(point.z)
        x_cords.append(x_row)
        y_cords.append(y_row)
        z_cords.append(z_row)
    return np.array(x_cords), np.array(y_cords), np.array(z_cords)