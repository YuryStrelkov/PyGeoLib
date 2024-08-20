from ...common import NUMERICAL_ACCURACY
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from matplotlib import pyplot as plt
from typing import Tuple, List
from .shape import Shape

# https://www.scratchapixel.com/lessons/geometry/bezier-curve-rendering-utah-teapot/bezier-curve.html


def cubic_bezier_patch_point(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3,
                             p5: Vector3, p6: Vector3, p7: Vector3, p8: Vector3,
                             p9: Vector3, p10: Vector3, p11: Vector3, p12: Vector3,
                             p13: Vector3, p14: Vector3, p15: Vector3, p16: Vector3,
                             u: float, v: float) -> Tuple[Vector3, Vector3]:
    phi1: float = (1.0 - u) * (1.0 - u) * (1.0 - u)
    phi4: float = u * u * u
    phi2: float = 3.0 * phi4 - 6.0 * u * u + 3.0 * u
    phi3: float = -3.0 * phi4 + 3.0 * u * u

    psi1: float = (1.0 - v) * (1.0 - v) * (1.0 - v)
    psi4: float = v * v * v
    psi2: float = 3.0 * psi4 - 6.0 * v * v + 3.0 * v
    psi3: float = -3.0 * psi4 + 3.0 * v * v

    p: Vector3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + p4 * phi1 * psi4 + \
                 p5 * phi2 * psi1 + p6 * phi2 * psi2 + p7 * phi2 * psi3 + p8 * phi2 * psi4 + \
                 p9 * phi3 * psi1 + p10 * phi3 * psi2 + p11 * phi3 * psi3 + p12 * phi3 * psi4 + \
                 p13 * phi4 * psi1 + p14 * phi4 * psi2 + p15 * phi4 * psi3 + p16 * phi4 * psi4

    d4: float = 3.0 * u * u
    d1: float = -3.0 + 6.0 * u - d4
    d2: float = 3.0 * phi4 - 12.0 * u + 3.0
    d3: float = -3.0 * phi4 + 6.0 * u

    dpu: Vector3 = p1 * d1 * psi1 + p2 * d1 * psi2 + p3 * d1 * psi3 + p4 * d1 * psi4 + \
                   p5 * d2 * psi1 + p6 * d2 * psi2 + p7 * d2 * psi3 + p8 * d2 * psi4 + \
                   p9 * d3 * psi1 + p10 * d3 * psi2 + p11 * d3 * psi3 + p12 * d3 * psi4 + \
                   p13 * d4 * psi1 + p14 * d4 * psi2 + p15 * d4 * psi3 + p16 * d4 * psi4

    d4 = 3.0 * v * v
    d1 = -3.0 + 6.0 * v - d4
    d2 = 3.0 * phi4 - 12.0 * v + 3.0
    d3 = -3.0 * phi4 + 6.0 * v

    dpv: Vector3 = p1 * phi1 * d1 + p2 * phi1 * d2 + p3 * phi1 * d3 + p4 * phi1 * d4 + \
                   p5 * phi2 * d1 + p6 * phi2 * d2 + p7 * phi2 * d3 + p8 * phi2 * d4 + \
                   p9 * phi3 * d1 + p10 * phi3 * d2 + p11 * phi3 * d3 + p12 * phi3 * d4 + \
                   p13 * phi4 * d1 + p14 * phi4 * d2 + p15 * phi4 * d3 + p16 * phi4 * d4

    return p, Vector3.cross(dpv, dpu).normalize()


def cubic_bezier_patch_position(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3,
                                p5: Vector3, p6: Vector3, p7: Vector3, p8: Vector3,
                                p9: Vector3, p10: Vector3, p11: Vector3, p12: Vector3,
                                p13: Vector3, p14: Vector3, p15: Vector3, p16: Vector3,
                                u: float, v: float) -> Vector3:
    phi1: float = (1.0 - u) * (1.0 - u) * (1.0 - u)
    phi4: float = u * u * u
    phi2: float = 3.0 * phi4 - 6.0 * u * u + 3.0 * u
    phi3: float = -3.0 * phi4 + 3.0 * u * u

    psi1: float = (1.0 - v) * (1.0 - v) * (1.0 - v)
    psi4: float = v * v * v
    psi2: float = 3.0 * psi4 - 6.0 * v * v + 3.0 * v
    psi3: float = -3.0 * psi4 + 3.0 * v * v

    p: Vector3 = p1 * phi1 * psi1 + p2 * phi1 * psi2 + p3 * phi1 * psi3 + p4 * phi1 * psi4 + \
                 p5 * phi2 * psi1 + p6 * phi2 * psi2 + p7 * phi2 * psi3 + p8 * phi2 * psi4 + \
                 p9 * phi3 * psi1 + p10 * phi3 * psi2 + p11 * phi3 * psi3 + p12 * phi3 * psi4 + \
                 p13 * phi4 * psi1 + p14 * phi4 * psi2 + p15 * phi4 * psi3 + p16 * phi4 * psi4
    return p


def cubic_bezier_patch_normal(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3,
                              p5: Vector3, p6: Vector3, p7: Vector3, p8: Vector3,
                              p9: Vector3, p10: Vector3, p11: Vector3, p12: Vector3,
                              p13: Vector3, p14: Vector3, p15: Vector3, p16: Vector3,
                              u: float, v: float) -> Vector3:
    phi1: float = (1.0 - u) * (1.0 - u) * (1.0 - u)
    phi4: float = u * u * u
    phi2: float = 3.0 * phi4 - 6.0 * u * u + 3.0 * u
    phi3: float = -3.0 * phi4 + 3.0 * u * u

    psi1: float = (1.0 - v) * (1.0 - v) * (1.0 - v)
    psi4: float = v * v * v
    psi2: float = 3.0 * psi4 - 6.0 * v * v + 3.0 * v
    psi3: float = -3.0 * psi4 + 3.0 * v * v

    d4: float = 3.0 * u * u
    d1: float = -3.0 + 6.0 * u - d4
    d2: float = 3.0 * phi4 - 12.0 * u + 3.0
    d3: float = -3.0 * phi4 + 6.0 * u

    dpu: Vector3 = p1 * d1 * psi1 + p2 * d1 * psi2 + p3 * d1 * psi3 + p4 * d1 * psi4 + \
                   p5 * d2 * psi1 + p6 * d2 * psi2 + p7 * d2 * psi3 + p8 * d2 * psi4 + \
                   p9 * d3 * psi1 + p10 * d3 * psi2 + p11 * d3 * psi3 + p12 * d3 * psi4 + \
                   p13 * d4 * psi1 + p14 * d4 * psi2 + p15 * d4 * psi3 + p16 * d4 * psi4

    d4 = 3.0 * v * v
    d1 = -3.0 + 6.0 * v - d4
    d2 = 3.0 * phi4 - 12.0 * v + 3.0
    d3 = -3.0 * phi4 + 6.0 * v

    dpv: Vector3 = p1 * phi1 * d1 + p2 * phi1 * d2 + p3 * phi1 * d3 + p4 * phi1 * d4 + \
                   p5 * phi2 * d1 + p6 * phi2 * d2 + p7 * phi2 * d3 + p8 * phi2 * d4 + \
                   p9 * phi3 * d1 + p10 * phi3 * d2 + p11 * phi3 * d3 + p12 * phi3 * d4 + \
                   p13 * phi4 * d1 + p14 * phi4 * d2 + p15 * phi4 * d3 + p16 * phi4 * d4

    return Vector3.cross(dpv, dpu).normalize()


PATCH_CONTROL_POINTS = (Vector3(-0.5, 0, -0.5),
                        Vector3(-0.1666, 0.0, -0.5),
                        Vector3(0.1666, 0.0, -0.5),
                        Vector3(0.5, 0.0, -0.5),
                        Vector3(-0.5, 0.0, -0.1666),
                        Vector3(-0.1666, 0.0, -0.1666),
                        Vector3(0.1666, 0.0, -0.1666),
                        Vector3(0.5, 0.0, -0.1666),
                        Vector3(-0.5, 0.0, 0.1666),
                        Vector3(-0.1666, 0.0, 0.1666),
                        Vector3(0.1666, 0.0, 0.1666),
                        Vector3(0.5, 0.0, 0.1666),
                        Vector3(-0.5, 0.0, 0.5),
                        Vector3(-0.1666, 0.0, 0.5),
                        Vector3(0.1666, 0.0, 0.5),
                        Vector3(0.5, 0.0, 0.5))


class BezierPatch(Shape):
    def __init__(self, points: Tuple[Vector3, ...] = None):
        super().__init__()
        if points:
            self._controllers = [p for p, i in zip(points, range(16))]
            return
        self._controllers: List[Vector3] = list(PATCH_CONTROL_POINTS)

    def __str__(self):
        nl = ",\n\t\t"
        return f"{{\n" \
               f"\t\"origin\" :{self.transform.origin},\n" \
               f"\t\"scale\"  :{self.transform.scale},\n" \
               f"\t\"angles\" :{self.transform.angles},\n" \
               f"\t\"controllers\":\n\t[\n\t\t{nl.join(str(v) for v in self._controllers)}\n\t]\n}}"

    @property
    def control_points(self) -> List[Vector3]:
        return self._controllers

    def normal(self, uv: Vector2) -> Vector3:
        dpu = (self.point(Vector2(uv.x + NUMERICAL_ACCURACY, uv.y)) -
               self.point(Vector2(uv.x - NUMERICAL_ACCURACY, uv.y))).normalize()
        dpv = (self.point(Vector2(uv.x, uv.y + NUMERICAL_ACCURACY)) -
               self.point(Vector2(uv.x, uv.y - NUMERICAL_ACCURACY))).normalize()
        return Vector3.cross(dpu, dpv).normalize()

    def point(self, uv: Vector2) -> Vector3:
        return cubic_bezier_patch_position(*self._controllers, *uv)

    def surface_orientation(self) -> float:
        return 1.0

    def draw_shape_gizmos(self, axis=None):
        axis = axis if axis else plt.axes(projection='3d')
        pts = self._controllers
        for i in range(4):
            p1, p2, p3, p4 = pts[i * 4: (i + 1) * 4]
            axis.plot((p1.x, p2.x, p3.x, p4.x),
                      (p1.y, p2.y, p3.y, p4.y),
                      (p1.z, p2.z, p3.z, p4.z), ':k')
            p1, p2, p3, p4 = pts[i * 4 ], pts[i * 4 + 1], pts[i * 4 + 2], pts[i * 4 + 3]
            axis.plot((p1.x,), (p1.y,), (p1.z,), 'or')
            axis.plot((p2.x,), (p2.y,), (p2.z,), 'or')
            axis.plot((p3.x,), (p3.y,), (p3.z,), 'or')
            axis.plot((p4.x,), (p4.y,), (p4.z,), 'or')
        return axis
