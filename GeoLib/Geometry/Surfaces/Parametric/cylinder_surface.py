from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from ...common import TWO_PI
from math import sin, cos
from .parametric_surface import ParametricSurface


class CylinderSurface(ParametricSurface):
    def __init__(self, radius1: float = 1.0, radius2: float = 1.0, height: float = 1.0,
                 angle: float = TWO_PI, start_angle: float = 0.0):
        super().__init__()
        self.radius1 = radius1
        self.radius2 = radius2
        self.height = height
        self.angle = angle
        self.start_angle = start_angle

    @property
    def radius1(self) -> float:
        return self._radius1

    @radius1.setter
    def radius1(self, value: float) -> None:
        self._radius1 = float(value)

    @property
    def radius2(self) -> float:
        return self._radius2

    @radius2.setter
    def radius2(self, value: float) -> None:
        self._radius2 = float(value)

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, value: float) -> None:
        self._angle = float(value)
        self.inner_oriented = self.angle > 0.0

    @property
    def start_angle(self) -> float:
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: float) -> None:
        self._start_angle = float(value)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = float(value)

    def __str__(self):
        return f"{{\n" \
               f"\t\"origin\"  :{self.transform.origin},\n" \
               f"\t\"scale\"   :{self.transform.scale},\n" \
               f"\t\"angles\"  :{self.transform.angles},\n" \
               f"\t\"radius1\" :{self.radius1},\n" \
               f"\t\"radius2\" :{self.radius2},\n" \
               f"\t\"angle\"   :{self.angle},\n" \
               f"\t\"height\"  :{self.height}\n" \
               f"}}"

    # def surface_orientation(self) -> float:
    #     return 1.0 if self.angle > 0.0 else -1.0

    @property
    def length(self) -> float:
        return abs(TWO_PI * (self.radius2 + self.radius1) * 0.5 * self.angle)

    def point(self, uv: Vector2) -> Vector3:
        u, v = uv
        u *= self.angle
        u -= self.start_angle
        radius = self.radius1 + (self.radius2 - self.radius1) * v
        xs = cos(u) * radius
        ys = sin(u) * radius
        zs = self.height * v
        return Vector3(xs, ys, zs)
