from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from math import sin, cos, sqrt
from ...common import TWO_PI
from .shape import Shape


class Helix(Shape):
    def __init__(self, radius1: float = 1.0, radius2: float = 1.0, height: float = 1.0, turns: float = 4):
        super().__init__()
        self.radius1 = radius1
        self.radius2 = radius2
        self.height = height
        self.turns = turns

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
    def turns(self) -> float:
        return self._turns

    @turns.setter
    def turns(self, value: float) -> None:
        self._turns = float(value)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = float(value)

    def __str__(self):
        return f"{{\n" \
               f"\t\"origin\" :{self.transform.origin},\n" \
               f"\t\"scale\"  :{self.transform.scale},\n" \
               f"\t\"angles\" :{self.transform.angles},\n" \
               f"\t\"radius\" :{self.radius1},\n" \
               f"\t\"radius\" :{self.radius2},\n" \
               f"\t\"radius\" :{self.height},\n" \
               f"\t\"turns\"  :{self.turns}\n" \
               f"}}"

    def surface_orientation(self) -> float:
        return 1.0 if self.turns > 0.0 else -1.0

    @property
    def length(self) -> float:
        return sqrt((TWO_PI * self.radius2 * self.turns) ** 2 + (self.height * self.turns) ** 2)

    def normal(self, uv: Vector2) -> Vector3:
        u, v = uv
        u *= TWO_PI * self.turns
        v *= TWO_PI
        # du( cos(u) * (r * cos(v) + sx * 0.5))     = (-sin(u) * (r * cos(v) + sx * 0.5))
        # du( sin(u) * (r * cos(v) + sy * 0.5))     = ( cos(u) * (r * cos(v) + sy * 0.5))
        # du( r * sin(v) + u / (2.0 * pi * t) * sz) = 1 / (2.0 * pi * t) * sz
        du = Vector3(-sin(u) * (self.radius1 * cos(v) + self.radius2 * 0.5),
                      cos(u) * (self.radius1 * cos(v) + self.radius2 * 0.5),
                      1.0 / (TWO_PI * self.turns) * self.height)
        # dv( cos(u) * (r * cos(v) + sx * 0.5))     = -cos(u) * r * cos(v)
        # dv( sin(u) * (r * cos(v) + sy * 0.5))     = -sin(u) * r * cos(v)
        # dv( r * sin(v) + u / (2.0 * pi * t) * sz) = r * cos(v)
        dv = Vector3(-cos(u) * sin(v) * self.radius1,
                     -sin(u) * sin(v) * self.radius1,
                     cos(v) * self.radius1)
        return Vector3.cross(dv, du).normalize()

    def point(self, uv: Vector2) -> Vector3:
        u, v = uv
        u *= TWO_PI * self.turns
        v *= TWO_PI
        xs = cos(u) * (self.radius1 * cos(v) + self.radius2 * 0.5) - self.radius2 * 0.5
        ys = sin(u) * (self.radius1 * cos(v) + self.radius2 * 0.5)
        zs = self.radius1 * sin(v) + u / (TWO_PI * self.turns) * self.height
        return Vector3(xs, ys, zs)
