from ...Shapes.Curves.bezier import bezier_interpolate_pt_3d
from ...Transformations.quaternion import Quaternion
from ...Vectors.vector2 import Vector2
from ...Vectors.vector3 import Vector3
from ...common import TWO_PI
from .shape import Shape
from typing import List, Tuple, Union


class LatheShape(Shape):
    def __init__(self, points: Union[List[Vector3], Tuple[Vector3, ...]]):
        super().__init__()
        self._profile_shape = list(points)
        self.interpolation_mode = 1
        self._start_angle = 0.0
        self._lathe_angle = TWO_PI
        self._closed = False
        self._axis_offset = 0.0
        self._interpolate_mode = 1
        self._axis = Vector3(0.0, 0.0, 1.0)
        self._right = sum(Vector3.cross(a, b) for a, b in zip(self._profile_shape[:-1], self._profile_shape[1:]))
        self._right = Vector3.cross(self._axis, self._right.normalize()).normalize()

    @property
    def axis_offset(self) -> float:
        return self._axis_offset

    @axis_offset.setter
    def axis_offset(self, value: float) -> None:
        self._axis_offset = float(value)

    @property
    def lathe_angle(self) -> float:
        return self._lathe_angle

    @lathe_angle.setter
    def lathe_angle(self, value: float) -> None:
        self._lathe_angle = float(value)

    @property
    def start_angle(self) -> float:
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: float) -> None:
        self._start_angle = float(value)

    @property
    def axis(self) -> Vector3:
        return self._axis

    @axis.setter
    def axis(self, value: Vector3) -> None:
        self._axis = value.normalized

    @property
    def interpolation_mode(self) -> int:
        return self._interpolate_mode

    @interpolation_mode.setter
    def interpolation_mode(self, value: int) -> None:
        self._interpolate_mode = int(value) if int(value) in (0, 1) else self._interpolate_mode

    def point(self, uv: Vector2) -> Vector3:
        position = bezier_interpolate_pt_3d(uv.x, self._profile_shape) + self.axis_offset * self._right
        rotation = Quaternion.from_axis_and_angle(self.axis, uv.y * self.lathe_angle + self.start_angle)
        return self.transform.transform_vect(rotation.rotate(position))
