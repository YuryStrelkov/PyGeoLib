from ..BezierShapes import utah_teapot_shape, box_shape, sphere_shape, spring_shape, DRAW_NORMALS, cylinder_shape
from matplotlib import pyplot as plt


def bezier_path_example():
    teapot_shapes = cylinder_shape()  # utah_teapot_shape()
    teapot_shapes.resolution = (2, 32)
    axis = None
    for shape in (teapot_shapes, ):
        axis = shape.draw_shape(axis, DRAW_NORMALS)
    axis.set_aspect('equal', 'box')
    plt.show()


code = """
def bezier_path_example():
    teapot_shapes = utah_teapot_shape()
    axis = None
    for shape in teapot_shapes:
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()
"""


if __name__ == "__main__":
    print(code)
    bezier_path_example()
