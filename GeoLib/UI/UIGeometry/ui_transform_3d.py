from PyQt5.QtWidgets import QVBoxLayout, QFrame, QDoubleSpinBox
from .geometry_ui_common import TRANSFORM_3D_LABELS
from .ui_vector_3 import UIVector3
from Geometry import Transform3d
from typing import Union, Tuple


class UITransform3d(QFrame):
    __slots__ = ('_items', '_transform')

    def __init__(self):
        super().__init__()
        items = []
        layout = QVBoxLayout()
        layout.addStretch()
        for label in TRANSFORM_3D_LABELS:
            vector = UIVector3(label)
            layout.addWidget(vector)
            items.append(vector)
        self._items: Tuple[QDoubleSpinBox, ...] = tuple(items)
        self._transform: Union[Transform3d, None] = None
        self.setLayout(layout)

    @property
    def origin(self) -> UIVector3:
        return self._items[0]

    @property
    def scale(self) -> UIVector3:
        return self._items[1]

    @property
    def angles(self) -> UIVector3:
        return self._items[2]

    @property
    def transform(self) -> Transform3d:
        return self._transform

    def _orig_x(self, x: float) -> None:
        self._transform.x = x

    def _orig_y(self, y: float) -> None:
        self._transform.y = y

    def _orig_z(self, z: float) -> None:
        self._transform.z = z

    def _scl_x(self, x: float) -> None:
        self._transform.sx = x

    def _scl_y(self, y: float) -> None:
        self._transform.sy = y

    def _scl_z(self, z: float) -> None:
        self._transform.sz = z

    def _ang_x(self, x: float) -> None:
        self._transform.ax = x

    def _ang_y(self, y: float) -> None:
        self._transform.ay = y

    def _ang_z(self, z: float) -> None:
        self._transform.az = z

    def clear_callbacks(self):
        self.origin.clear_callbacks()
        self.scale.clear_callbacks()
        self.angles.clear_callbacks()

    @transform.setter
    def transform(self, value: Union[Transform3d, None]) -> None:
        self._transform = value
        if not self._transform:
            self.clear_callbacks()
            return
        self.origin.on_x_change(lambda v: self._orig_x(v))
        self.origin.on_y_change(lambda v: self._orig_y(v))
        self.origin.on_y_change(lambda v: self._orig_y(v))

        self.scale.on_x_change(lambda v: self._scl_x(v))
        self.scale.on_y_change(lambda v: self._scl_y(v))
        self.scale.on_y_change(lambda v: self._scl_z(v))

        self.angles.on_x_change(lambda v: self._ang_x(v))
        self.angles.on_y_change(lambda v: self._ang_y(v))
        self.angles.on_y_change(lambda v: self._ang_z(v))
