from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QApplication, QSizePolicy
from .UICollapsible import CollapsibleBox
from .UIGeometry import UITransform3d
from .UIGeometry import UIVector3
from .UIGeometry import UIMatrix3
import sys


class UIExample(QMainWindow):
    def _build_transform_collapsable(self):
        self._transform = UITransform3d()
        box = CollapsibleBox()
        box_layout = QVBoxLayout()
        box_layout.addWidget(self._transform)
        box.set_content_layout(box_layout)
        return box

    def _build_surface_m_collapsable(self):
        self._surface_b = UIMatrix3("Matrix3x3")
        self._surface_k = UIVector3("Vector3")
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
        layout.addStretch()
        container.setLayout(layout)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setWindowTitle("My App")
        self.setCentralWidget(container)

    @classmethod
    def create_and_run(cls, style_sheets_path: str = "ui/app-style.CSS"):
        app = QApplication(sys.argv)
        try:
            with open(style_sheets_path, 'rt') as input_css:
                app.setStyleSheet("".join(line for line in input_css))
        except FileNotFoundError as ex:
            print(f"no style file at path \"{style_sheets_path}\"\n{ex.args}")
        window = cls()
        window.show()
        app.exec()
