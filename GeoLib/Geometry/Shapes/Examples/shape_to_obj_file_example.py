from ..BezierShapes.custom_shapes import utah_teapot_shape
from ...Meshes.obj_file_builder import build_obj_file
from matplotlib import pyplot as plt


def shape_to_obj_file_example():
    teapot_shapes = utah_teapot_shape()
    axis = None
    with open('teapot.obj', 'wt') as teapot:
        build_obj_file(teapot_shapes, teapot)

    with open('teapot_triangulated.obj', 'wt') as teapot:
        build_obj_file(teapot_shapes, teapot, True)

    for shape in teapot_shapes:
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()


code = """
def shape_to_obj_file_example():
    teapot_shapes = utah_teapot_shape()
    axis = None
    with open('teapot.obj') as teapot:
        build_obj_file(teapot_shapes, teapot)

    with open('teapot_triangulated.obj') as teapot:
        build_obj_file(teapot_shapes, teapot, True)

    for shape in teapot_shapes:
        axis = shape.draw_shape(axis)
    axis.set_aspect('equal', 'box')
    plt.show()
"""


if __name__ == "__main__":
    print(code)
    shape_to_obj_file_example()
