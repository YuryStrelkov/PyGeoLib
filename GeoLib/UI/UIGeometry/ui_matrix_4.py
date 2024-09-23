from .geometry_ui_common import MATRIX_4_LABELS
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QFrame
from .ui_vector_4 import UIVector4


class UIMatrix4(QFrame):
    __slots__ = ('_items', )

    def __init__(self, name: str = None):
        super().__init__()
        layout = QVBoxLayout()
        layout.addStretch()
        if name:
            layout.addWidget(QLabel(name))
        self._items = (UIVector4(None, MATRIX_4_LABELS[0:4]),
                       UIVector4(None, MATRIX_4_LABELS[4:8]),
                       UIVector4(None, MATRIX_4_LABELS[8:12]),
                       UIVector4(None, MATRIX_4_LABELS[12:16]))
        for item in self._items:
            layout.addWidget(item)
        self.setLayout(layout)

    @property
    def row1(self) -> UIVector4:
        return self._items[0]

    @property
    def row2(self) -> UIVector4:
        return self._items[1]

    @property
    def row3(self) -> UIVector4:
        return self._items[2]

    @property
    def row4(self) -> UIVector4:
        return self._items[2]
