from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from ...common import TWO_PI
from math import sin, cos
from .parametric_surface import ParametricSurface


class TorusSurface(ParametricSurface):
    def __init__(self, radius1: float = 1.0, radius2: float = 1.0, turns: float = 1.0):
        super().__init__()
        self.radius1 = radius1
        self.radius2 = radius2
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
        self.inner_oriented =  self.turns > 0.0

    def __str__(self):
        return f"{{\n" \
               f"\t\"origin\"  :{self.transform.origin},\n" \
               f"\t\"scale\"   :{self.transform.scale},\n" \
               f"\t\"angles\"  :{self.transform.angles},\n" \
               f"\t\"radius1\" :{self.radius1},\n" \
               f"\t\"radius2\" :{self.radius2},\n" \
               f"\t\"param\"   :{self.turns}\n" \
               f"}}"

    # def surface_orientation(self) -> float:
    #     return 1.0 if self.turns > 0.0 else -1.0

    @property
    def length(self) -> float:
        return abs(TWO_PI * self.radius2 * self.turns)

    def point(self, uv: Vector2) -> Vector3:
        u, v = uv
        u *= TWO_PI * self.turns
        v *= TWO_PI
        xs = cos(u) * (self.radius1 * cos(v) + self.radius2 * 0.5) - self.radius2 * 0.5
        ys = sin(u) * (self.radius1 * cos(v) + self.radius2 * 0.5)
        zs = self.radius1 * sin(v)
        return Vector3(xs, ys, zs)

    def normal(self, uv: Vector2) -> Vector3:
        u, v = uv
        u *= TWO_PI * self.turns
        v *= TWO_PI
        du = Vector3(-sin(u) * self.radius1 * (self.radius1 * cos(v) + self.radius2 * 0.5),
                     cos(u) * self.radius1 * (self.radius1 * cos(v) + self.radius2 * 0.5),
                     0.0)
        dv = Vector3(-cos(u) * self.radius1 * sin(v),
                     -sin(u) * self.radius1 * sin(v),
                     self.radius1 * cos(v))
        return Vector3.cross(dv, du).normalize()

