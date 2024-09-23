from .geometry_ui_common import VECTOR_3_LABELS
from PyQt5.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QFrame
from typing import Tuple, Callable
import contextlib


class UIVector3(QFrame):
    __slots__ = ('_items', )

    def __init__(self, name: str = None, labels: Tuple[str, ...] = None):
        super().__init__()
        layout = QHBoxLayout()
        layout.addStretch()
        if name:
            layout.addWidget(QLabel(name))
        if labels:
            assert (len(labels) == 3)
        else:
            labels = VECTOR_3_LABELS
        self._items: Tuple[QDoubleSpinBox, ...] = tuple(QDoubleSpinBox() for _ in range(3))
        for label, item in zip(labels, self._items):
            item.setMinimum(-1e32)
            item.setMaximum( 1e32)
            item.setSingleStep(0.5)
            item.setPrefix(label)
            layout.addWidget(item)
        self.setLayout(layout)

    def clear_callbacks(self):
        with contextlib.suppress(TypeError):
            self._items[0].valueChanged.disconnect()
        with contextlib.suppress(TypeError):
            self._items[1].valueChanged.disconnect()
        with contextlib.suppress(TypeError):
            self._items[2].valueChanged.disconnect()

    def on_x_change(self, callback: Callable[[float], None]):
        with contextlib.suppress(TypeError):
            self._items[0].valueChanged.disconnect()
        self._items[0].valueChanged.connect(lambda v: callback(v))

    def on_y_change(self, callback: Callable[[float], None]):
        with contextlib.suppress(TypeError):
            self._items[1].valueChanged.disconnect()
        self._items[1].valueChanged.connect(lambda v: callback(v))

    def on_z_change(self, callback: Callable[[float], None]):
        with contextlib.suppress(TypeError):
            self._items[2].valueChanged.disconnect()
        self._items[2].valueChanged.connect(lambda v: callback(v))
