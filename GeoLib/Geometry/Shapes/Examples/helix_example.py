from matplotlib import pyplot as plt
from ..BezierShapes import Helix
from ...Vectors import Vector3


def helix_example():
    helix1 = Helix(0.05, 1.0, 0.25, 2)
    helix1.resolution = (512, 16)
    helix1.transform.origin = Vector3(0, -1.5, 0)
    helix2 = Helix(0.05, 1.5, 0.5, 3)
    helix2.resolution = (512, 16)
    helix3 = Helix(0.05, 2.0, 1.0, 5)
    helix3.resolution = (512, 16)
    helix3.transform.origin = Vector3(0, 2.125, 0)
    axis = None
    for shape in (helix1, helix2, helix3):
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()


code = """
def helix_example():
    helix1 = Helix(0.05, 1.0, 0.25, 2)
    helix1.resolution = (512, 16)
    helix1.transform.origin = Vector3(0, -1.5, 0)
    helix2 = Helix(0.05, 1.5, 0.5, 3)
    helix2.resolution = (512, 16)
    helix3 = Helix(0.05, 2.0, 1.0, 5)
    helix3.resolution = (512, 16)
    helix3.transform.origin = Vector3(0, 2.125, 0)
    axis = None
    for shape in (helix1, helix2, helix3):
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()
"""


if __name__ == "__main__":
    print(code)
    helix_example()
