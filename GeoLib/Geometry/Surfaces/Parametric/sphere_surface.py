from ...common import TWO_PI, PI
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from math import sin, cos
from .parametric_surface import ParametricSurface


class SphereSurface(ParametricSurface):
    def __init__(self, radius: float = 1.0, uv0: Vector2 = None, uv1: Vector2 = None):
        super().__init__()
        self.radius = radius
        self.uv0 = uv0 if uv0 else Vector2(0.0, 0.0)
        self.uv1 = uv1 if uv1 else Vector2(1.0, 1.0)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def uv0(self) -> Vector2:
        return self._uv0

    @property
    def uv1(self) -> Vector2:
        return self._uv1

    @uv0.setter
    def uv0(self, value: Vector2) -> None:
        self._uv0 = value

    @uv1.setter
    def uv1(self, value: Vector2) -> None:
        self._uv1 = value

    def __str__(self):
        return f"{{\n" \
               f"\t\"origin\":{self.transform.origin},\n" \
               f"\t\"scale\" :{self.transform.scale},\n" \
               f"\t\"angles\":{self.transform.angles},\n" \
               f"\t\"radius\":{self.radius},\n" \
               f"\t\"uv0\"   :{self.uv0},\n" \
               f"\t\"uv1\"   :{self.uv1}\n" \
               f"}}"

    def point(self, uv: Vector2) -> Vector3:
        u = TWO_PI * (self.uv0.x + uv.x * (self.uv1.x - self.uv0.x))
        v = PI * (self.uv0.y + uv.y * (self.uv1.y - self.uv0.y))
        cos_u, cos_v = cos(u), cos(v)
        sin_u, sin_v = sin(u), sin(v)
        x = self.radius * sin_v * cos_u
        y = self.radius * sin_v * sin_u
        z = self.radius * cos_v
        return Vector3(x, y, z)

    def normal(self, uv: Vector2) -> Vector3:
        return self.point(uv).normalize()

