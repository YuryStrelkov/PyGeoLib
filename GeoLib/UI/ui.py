import contextlib
from typing import Tuple, Callable, Union

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,\
    QMainWindow, QVBoxLayout, QDoubleSpinBox, QHBoxLayout, QLabel, QFrame
import sys # Только для доступа к аргументам командной строки

from Geometry import Transform3d
from UICollapsible import CollapsibleBox


class QtVector2(QFrame):
    _labels = ('x: ', 'y: ')
    __slots__ = ('_items', )

    def __init__(self, name: str = None, labels: Tuple[str, ...] = None):
        super().__init__()
        layout = QHBoxLayout()
        if name:
            layout.addWidget(QLabel(name))
        if labels:
            assert (len(labels) == 2)
        else:
            labels = QtVector2._labels
        self._items: Tuple[QDoubleSpinBox, ...] = tuple(QDoubleSpinBox() for _ in range(2))
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

    def on_x_change(self, callback: Callable[[float], None]):
        with contextlib.suppress(TypeError):
            self._items[0].valueChanged.disconnect()
        self._items[0].valueChanged.connect(lambda v: callback(v))

    def on_y_change(self, callback: Callable[[float], None]):
        with contextlib.suppress(TypeError):
            self._items[1].valueChanged.disconnect()
        self._items[1].valueChanged.connect(lambda v: callback(v))


class QtVector3(QFrame):
    _labels = ('x: ', 'y: ', 'z: ')
    __slots__ = ('_items', )

    def __init__(self, name: str = None, labels: Tuple[str, ...] = None):
        super().__init__()
        layout = QHBoxLayout()
        if name:
            layout.addWidget(QLabel(name))
        if labels:
            assert (len(labels) == 3)
        else:
            labels = QtVector3._labels
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


class QtVector4(QFrame):
    _labels = ('x: ', 'y: ', 'z: ', 'w: ')
    __slots__ = ('_items', )

    def __init__(self, name: str = None, labels: Tuple[str, ...] = None):
        super().__init__()
        layout = QHBoxLayout()
        if name:
            layout.addWidget(QLabel(name))
        if labels:
            assert (len(labels) == 4)
        else:
            labels = QtVector4._labels
        self._items: Tuple[QDoubleSpinBox, ...] = tuple(QDoubleSpinBox() for _ in range(4))
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
        with contextlib.suppress(TypeError):
            self._items[3].valueChanged.disconnect()

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

    def on_w_change(self, callback: Callable[[float], None]):
        with contextlib.suppress(TypeError):
            self._items[3].valueChanged.disconnect()
        self._items[3].valueChanged.connect(lambda v: callback(v))


class QtMatrix3(QFrame):
    __slots__ = ('_items', )

    def __init__(self, name: str = None):
        super().__init__()
        layout = QVBoxLayout()
        if name:
            layout.addWidget(QLabel(name))
        self._items = (QtVector3(None, ('m00: ', 'm02: ', 'm02: ')),
                       QtVector3(None, ('m10: ', 'm12: ', 'm12: ')),
                       QtVector3(None, ('m20: ', 'm22: ', 'm22: ')))
        for item in self._items:
            layout.addWidget(item)
        self.setLayout(layout)

    def row1(self) -> QtVector3:
        return self._items[0]

    def row2(self) -> QtVector3:
        return self._items[1]

    def row3(self) -> QtVector3:
        return self._items[2]


class QtMatrix4(QFrame):
    __slots__ = ('_items', )

    def __init__(self, name: str = None):
        super().__init__()
        layout = QVBoxLayout()
        if name:
            layout.addWidget(QLabel(name))
        self._items = (QtVector4(None, ('m00: ', 'm02: ', 'm02: ', 'm03: ')),
                       QtVector4(None, ('m10: ', 'm12: ', 'm12: ', 'm13: ')),
                       QtVector4(None, ('m20: ', 'm22: ', 'm22: ', 'm23: ')),
                       QtVector4(None, ('m30: ', 'm32: ', 'm32: ', 'm33: ')))
        for item in self._items:
            layout.addWidget(item)
        self.setLayout(layout)

    def row1(self) -> QtVector4:
        return self._items[0]

    def row2(self) -> QtVector4:
        return self._items[1]

    def row3(self) -> QtVector4:
        return self._items[2]

    def row4(self) -> QtVector4:
        return self._items[2]


class QtTransform3d(QFrame):
    _labels = ("Position:", "Scale:   ", "Rotation:")
    __slots__ = ('_items', '_transform')

    def __init__(self):
        super().__init__()
        items = []
        layout = QVBoxLayout()
        for label in QtTransform3d._labels:
            vector = QtVector3(label)
            layout.addWidget(vector)
            items.append(vector)
        self._items: Tuple[QDoubleSpinBox, ...] = tuple(items)
        self._transform: Union[Transform3d, None] = None
        self.setLayout(layout)

    @property
    def qt_origin(self) -> QtVector3:
        return self._items[0]

    @property
    def qt_scale(self) -> QtVector3:
        return self._items[1]

    @property
    def qt_angles(self) -> QtVector3:
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
        self.qt_origin.clear_callbacks()
        self.qt_scale.clear_callbacks()
        self.qt_angles.clear_callbacks()

    @transform.setter
    def transform(self, value: Union[Transform3d, None]) -> None:
        self._transform = value
        if not self._transform:
            self.clear_callbacks()
            return
        self.qt_origin.on_x_change(lambda v: self._orig_x(v))
        self.qt_origin.on_y_change(lambda v: self._orig_y(v))
        self.qt_origin.on_y_change(lambda v: self._orig_y(v))

        self.qt_scale.on_x_change(lambda v: self._scl_x(v))
        self.qt_scale.on_y_change(lambda v: self._scl_y(v))
        self.qt_scale.on_y_change(lambda v: self._scl_z(v))

        self.qt_angles.on_x_change(lambda v: self._ang_x(v))
        self.qt_angles.on_y_change(lambda v: self._ang_y(v))
        self.qt_angles.on_y_change(lambda v: self._ang_z(v))


class MainWindow(QMainWindow):

    def _build_transform_collapsable(self):
        self._transform = QtTransform3d()
        box = CollapsibleBox()
        box_layout = QVBoxLayout()
        box_layout.addWidget(self._transform)
        box.set_content_layout(box_layout)
        return box

    def _build_surface_m_collapsable(self):
        self._surface_b = QtMatrix3("Matrix3x3")
        self._surface_k = QtVector3("Vector3")
        box = CollapsibleBox()
        box_layout = QVBoxLayout()
        box_layout.addWidget(self._surface_b)
        box_layout.addWidget(self._surface_k)
        box.set_content_layout(box_layout)
        return box

    def __init__(self):
        super().__init__()
        self._transform = None
        self._surface_m = None
        self._surface_b = None
        self._surface_k = None
        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self._build_transform_collapsable())
        layout.addWidget(self._build_surface_m_collapsable())
        container.setLayout(layout)
        self.setWindowTitle("My App")
        self.setCentralWidget(container)


def application():
    app = QApplication(sys.argv)
    with open("app-style.css", 'rt') as input_css:
        app.setStyleSheet("".join(line for line in input_css))
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    application()