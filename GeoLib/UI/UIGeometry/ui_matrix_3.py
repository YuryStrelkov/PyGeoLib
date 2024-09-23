from .geometry_ui_common import MATRIX_3_LABELS
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QFrame
from .ui_vector_3 import UIVector3


class UIMatrix3(QFrame):
    __slots__ = ('_items', )

    def __init__(self, name: str = None):
        super().__init__()
        layout = QVBoxLayout()
        layout.addStretch()
        if name:
            layout.addWidget(QLabel(name))
        self._items = (UIVector3(None, MATRIX_3_LABELS[0:3]),
                       UIVector3(None, MATRIX_3_LABELS[3:6]),
                       UIVector3(None, MATRIX_3_LABELS[6:9]))
        for item in self._items:
            layout.addWidget(item)
        self.setLayout(layout)

    @property
    def row1(self) -> UIVector3:
        return self._items[0]

    @property
    def row2(self) -> UIVector3:
        return self._items[1]

    @property
    def row3(self) -> UIVector3:
        return self._items[2]