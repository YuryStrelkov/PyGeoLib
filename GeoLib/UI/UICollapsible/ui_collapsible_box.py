from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None, close_btn: bool = False):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(text=title, checkable=True, checked=True)
        # self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_pressed)
        self.toggle_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.toggle_button.setMinimumHeight(25)
        self.toggle_close = None
        if close_btn:
            self.toggle_close = QtWidgets.QToolButton(text='X', checkable=True, checked=True)
            self.toggle_close.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self.toggle_close.setStyleSheet("background-color: {}; color : white;".format(QColor(255, 0, 0).name()))
            self.toggle_close.setFixedHeight(22)
            self.toggle_close.setFixedWidth(22)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)
        self.content_area = QtWidgets.QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QtWidgets.QWidget()
        container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay = QtWidgets.QHBoxLayout(container)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        if self.toggle_close:
            lay.addWidget(self.toggle_close)

        lay1 = QtWidgets.QVBoxLayout(self)
        lay1.setSpacing(0)
        lay1.setContentsMargins(0, 0, 0, 0)
        lay1.addWidget(container)
        lay1.addWidget(self.content_area)

        self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self, b"minimumHeight"))
        self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self, b"maximumHeight"))
        self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self.content_area, b"maximumHeight"))

    @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow)
        self.toggle_animation.setDirection(QtCore.QAbstractAnimation.Forward if not checked else
                                           QtCore.QAbstractAnimation.Backward)
        self.toggle_animation.start()

    def set_content_layout(self, layout):
        curr_layout = self.content_area.layout()
        if curr_layout:
            curr_layout.deleteLater()
        self.content_area.setLayout(layout)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()
        for index in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(index)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


if __name__ == "__main__":
    import sys
    import random

    app = QtWidgets.QApplication(sys.argv)

    w = QtWidgets.QMainWindow()
    w.setCentralWidget(QtWidgets.QWidget())
    dock = QtWidgets.QDockWidget("Collapsible Demo")
    dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    scroll = QtWidgets.QScrollArea()
    dock.setWidget(scroll)
    content = QtWidgets.QWidget()
    scroll.setWidget(content)
    scroll.setWidgetResizable(True)
    vlay = QtWidgets.QVBoxLayout(content)
    for i in range(3):
        box = CollapsibleBox("Collapsible Box Header-{}".format(i))
        color = QtGui.QColor(*[random.randint(0, 255) for _ in range(3)])
        box.setStyleSheet("background-color: {}; color : white;".format(color.name()))

        vlay.addWidget(box)
        lay = QtWidgets.QVBoxLayout()
        for j in range(8):
            label = QtWidgets.QLabel("{}".format(j))
            color = QtGui.QColor(*[random.randint(0, 255) for _ in range(3)])
            label.setStyleSheet( "background-color: {}; color : white;".format(color.name()))
            label.setAlignment(QtCore.Qt.AlignCenter)
            lay.addWidget(label)

        box.set_content_layout(lay)
    vlay.addStretch()
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())