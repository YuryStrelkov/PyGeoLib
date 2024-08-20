from ..Vectors import Vector2
from typing import Tuple
import collections
import math

free_fall_acceleration_2d = Vector2(0, -9.81)

"""
#######################################################################################################################
#################################                 ZERO AIR RESISTANCE                 #################################
#######################################################################################################################
"""


class SurfaceParam(collections.namedtuple("SurfaceParam", "normal, height, absorption")):
    __slots__ = ()

    def __new__(cls, normal: Vector2 = None, height: float = 0.0, absorption: float = 0.0):
        return super(SurfaceParam, cls).__new__(cls, normal, height, absorption)


def ballistic_zero_resistance(p0: Vector2, v0: Vector2, time_val: float,
                              surface_args: SurfaceParam = None) -> Tuple[Vector2, Vector2]:
    vel = v0 + free_fall_acceleration_2d * time_val
    pos = p0 + v0 * time_val + free_fall_acceleration_2d * time_val * time_val * 0.5
    if surface_args:
        if pos.y < surface_args.height:
            pos.y = surface_args.height
            vel = Vector2.reflect(vel, surface_args.normal) * vel.magnitude * surface_args.absorption
    return vel, pos


def ballistic_time_range_zero_resistance(p0: Vector2, v0: Vector2, ground_level: float = 0.0):
    # |x|   |x0|   |vx|        |ax|
    # | | = |  | + |  | * t +  |  | * t * t * 0.5
    # |y|   |y0|   |vy|        |ay|
    #  0  =  y0  +  vy  * t +   ay * t * t * 0.5
    a = free_fall_acceleration_2d.y * 0.5
    b = v0.y
    c = p0.y - ground_level
    d = b * b - 4.0 * a * c
    if d < 0:
        return -1.0, -1.0
    d = math.sqrt(d)
    return (-b - d)  * 0.5 / a, (-b + d)  * 0.5 / a


# def _ode_ballistic_linear_resistance(p0: Vector2, v0: Vector2, t_steps: int = 1000) -> List[Vector2]:
#     t1, t2 = ballistic_time_range(p0, v0)
#     time_range = max(t1, t2)
#     dt = time_range / (t_steps - 1)
#     points   = []
#     velocity = Vector2(*v0)
#     position = Vector2(*p0)
#     points.append(Vector2(*position))
#     for index in range(t_steps):
#         velocity += free_fall_acceleration * dt  - 0.1 * velocity * velocity * dt
#         position += velocity * dt
#         if position.y < 0:
#             break
#             position.y = 0
#             velocity.y *= -1
#             velocity *= 0.125
#         points.append(Vector2(*position))
#     return points