from matplotlib import pyplot as plt
from ...Vectors import Vector3
from ..Parametric import TorusSurface


def torus_example():
    torus1 = TorusSurface(0.25, 1.0, 0.25)
    torus1.transform.origin = Vector3(0, 0, -0.75)
    torus2 = TorusSurface(0.25, 1.5, 0.5)
    torus3 = TorusSurface(0.25, 2.0, 1.0)
    torus3.transform.origin = Vector3(0, 0, 0.75)
    axis = None
    for shape in (torus1, torus2, torus3):
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()


code = """
def torus_example():
    torus1 = Torus(0.25, 1.0, 0.25)
    torus1.transform.origin = Vector3(0, 0, -0.75)
    torus2 = Torus(0.25, 1.5, 0.5)
    torus3 = Torus(0.25, 2.0, 1.0)
    torus3.transform.origin = Vector3(0, 0, 0.75)
    axis = None
    for shape in (torus1, torus2, torus3):
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()
"""


if __name__ == "__main__":
    print(code)
    torus_example()
