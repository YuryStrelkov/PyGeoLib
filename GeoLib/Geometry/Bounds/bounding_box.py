from ..Transformations.transform_3d import Transform3d
from ..Vectors.vector3 import Vector3
from typing import Generator
from ..common import *


class BoundingBox:

    __slots__ = "_max", "_min"

    def __init__(self):
        self._max: Vector3 = Vector3(NUMERICAL_MIN_VALUE, NUMERICAL_MIN_VALUE, NUMERICAL_MIN_VALUE)
        self._min: Vector3 = Vector3(NUMERICAL_MAX_VALUE, NUMERICAL_MAX_VALUE, NUMERICAL_MAX_VALUE)

    def __str__(self):
        return f"{{\n" \
               f"\t\"min\": {self.min},\n" \
               f"\t\"max\": {self.max}" \
               f"\n}}"

    @property
    def points(self) -> Generator[Vector3, None, None]:
        c = self.center
        s = 0.5 * self.size
        yield Vector3(c.x - s.x, c.y + s.y, c.z - s.z)
        yield Vector3(c.x + s.x, c.y - s.y, c.z - s.z)
        yield Vector3(c.x - s.x, c.y - s.y, c.z - s.z)
        yield Vector3(c.x + s.x, c.y + s.y, c.z - s.z)

        yield Vector3(c.x - s.x, c.y + s.y, c.z + s.z)
        yield Vector3(c.x + s.x, c.y - s.y, c.z + s.z)
        yield Vector3(c.x - s.x, c.y - s.y, c.z + s.z)
        yield Vector3(c.x + s.x, c.y + s.y, c.z + s.z)

    def reset(self) -> None:
        self._max = Vector3(NUMERICAL_MIN_VALUE, NUMERICAL_MIN_VALUE, NUMERICAL_MIN_VALUE)
        self._min = Vector3(NUMERICAL_MAX_VALUE, NUMERICAL_MAX_VALUE, NUMERICAL_MAX_VALUE)

    def contains(self, v: Vector3) -> bool:
        if any(lhs < rhs for lhs, rhs in zip(v, self.min)):
            return False
        if any(lhs > rhs for lhs, rhs in zip(v, self.max)):
            return False
        return True

    def encapsulate(self, v: Vector3) -> None:
        self._max = Vector3.max(self._max, v)
        self._min = Vector3.min(self._min, v)

    def encapsulate_bbox(self, v) -> None:
        self._max = Vector3.max(self._max, v.max)
        self._min = Vector3.min(self._min, v.min)

    def transform_bbox(self, transform: Transform3d) -> 'BoundingBox':
        bounds = BoundingBox()
        for pt in self.points:
            bounds.encapsulate(transform.transform_vect(pt))
        return bounds

    def inv_transform_bbox(self, transform: Transform3d) -> 'BoundingBox':
        bounds = BoundingBox()
        for pt in self.points:
            bounds.encapsulate(transform.inv_transform_vect(pt))
        return bounds

    @property
    def min(self) -> Vector3:
        return self._min

    @property
    def max(self) -> Vector3:
        return self._max

    @property
    def size(self) -> Vector3:
        return self._max - self._min

    @property
    def center(self) -> Vector3:
        return (self._max + self._min) * 0.5

    def sdf(self, point: Vector3) -> float:
        """
        Signed distance function
        @param point:
        @return:
        """
        orig = self.center
        size = self.size

        x_l = point.x - (orig.x - size.x * 0.5)
        x_r = point.x - (orig.x + size.x * 0.5)

        y_l = point.y - (orig.y - size.y * 0.5)
        y_r = point.y - (orig.y + size.y * 0.5)

        z_l = point.z - (orig.z - size.z * 0.5)
        z_r = point.z - (orig.z + size.z * 0.5)

        return max(max(max(abs(y_l), abs(y_r)) - size.y,
                       max(abs(x_l), abs(x_r)) - size.x),
                   max(abs(z_l), abs(z_r)) - size.z)
