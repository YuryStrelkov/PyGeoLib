from ..BezierShapes import LatheShape
from matplotlib import pyplot as plt
from ...Vectors import Vector3
from ...common import PI


def lathe_shape_example():
    points = (Vector3(0.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 3.0, 0.0), Vector3(0.0, 3.5, 1.0),
              Vector3(0.0, 3.5, 2.0), Vector3(0.0, 3.0, 3.0), Vector3(0.0, 2.0, 4.0), Vector3(0.0, 2.0, 5.0),
              Vector3(0.0, 2.0, 6.0), Vector3(0.0, 3.0, 6.0))

    shape = LatheShape(points)
    shape.resolution = (32, 32)
    shape.lathe_angle = PI * 0.5
    shape.axis_offset = 1.0
    axis = None
    for shape in (shape, ):
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()


code = """
    points = [Vector3(0.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 3.0, 0.0), Vector3(0.0, 3.5, 1.0),
              Vector3(0.0, 3.5, 2.0), Vector3(0.0, 3.0, 3.0), Vector3(0.0, 2.0, 4.0), Vector3(0.0, 2.0, 5.0),
              Vector3(0.0, 2.0, 6.0), Vector3(0.0, 3.0, 6.0)]
    shape = LatheShape(points)
    shape.resolution = (32, 32)
    axis = None
    for shape in (shape, ):
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()
"""


if __name__ == "__main__":
    print(code)
    lathe_shape_example()