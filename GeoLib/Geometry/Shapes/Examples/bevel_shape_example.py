from ..Curves import bezier_circle_3d, bezier_helics_3d, bezier_interpolate_pt_3d
from ..BezierShapes import BevelShape
from matplotlib import pyplot as plt
import numpy as np


def bevel_shape_bezier_interpolation_example():
    start_shape = bezier_circle_3d(radius=0.90, orientation=2)
    end_shape   = bezier_circle_3d(radius=0.25, orientation=2)
    shape_path  = bezier_helics_3d(radius=5.0, gap=1.5, turns=4.5, orientation=0)
    sweep_shape = BevelShape(start_shape, shape_path, end_shape)
    sweep_shape.resolution = (512, 32)
    axis = None
    for shape in (sweep_shape,):
        axis = shape.draw_shape(axis, 2)
    axis.set_aspect('equal', 'box')
    plt.show()


def bevel_shape_linear_interpolation_example():
    start_shape = bezier_circle_3d(radius=0.90, orientation=2)
    end_shape   = bezier_circle_3d(radius=0.25, orientation=2)
    shape_path  = bezier_helics_3d(radius=5.0, gap=1.5, turns=4.5, orientation=0)
    t_args1 = np.linspace(0.0, 1.0, 128)
    t_args2 = np.linspace(0.0, 1.0, 16)
    shape_path  = tuple(bezier_interpolate_pt_3d(ti, shape_path) for ti in t_args1.flat)
    start_shape = tuple(bezier_interpolate_pt_3d(ti, start_shape)for ti in t_args2.flat)
    end_shape   = tuple(bezier_interpolate_pt_3d(ti,   end_shape)for ti in t_args2.flat)
    sweep_shape = BevelShape(start_shape, shape_path, end_shape)
    sweep_shape.interpolation_mode = 0
    sweep_shape.resolution = (512, 32)
    axis = None
    for shape in (sweep_shape,):
        axis = shape.draw_shape(axis, 2)
    axis.set_aspect('equal', 'box')
    plt.show()


code = """
def bevel_shape_bezier_interpolation_example():
    start_shape = bezier_circle_3d(radius=0.90, orientation=2)
    end_shape   = bezier_circle_3d(radius=0.25, orientation=2)
    shape_path  = bezier_helics_3d(radius=5.0, gap=1.5, turns=4.5, orientation=0)
    sweep_shape = BevelShape(start_shape, shape_path, end_shape)
    sweep_shape.resolution = (512, 32)
    axis = None
    for shape in (sweep_shape,):
        axis = shape.draw_shape(axis, 2)
    axis.set_aspect('equal', 'box')
    plt.show()


def bevel_shape_linear_interpolation_example():
    start_shape = bezier_circle_3d(radius=0.90, orientation=2)
    end_shape   = bezier_circle_3d(radius=0.25, orientation=2)
    shape_path  = bezier_helics_3d(radius=5.0, gap=1.5, turns=4.5, orientation=0)
    t_args1 = np.linspace(0.0, 1.0, 128)
    t_args2 = np.linspace(0.0, 1.0, 16)
    shape_path  = tuple(bezier_interpolate_pt_3d(ti, shape_path) for ti in t_args1.flat)
    start_shape = tuple(bezier_interpolate_pt_3d(ti, start_shape)for ti in t_args2.flat)
    end_shape   = tuple(bezier_interpolate_pt_3d(ti,   end_shape)for ti in t_args2.flat)
    sweep_shape = BevelShape(start_shape, shape_path, end_shape)
    sweep_shape.interpolation_mode = 0
    sweep_shape.resolution = (512, 32)
    axis = None
    for shape in (sweep_shape,):
        axis = shape.draw_shape(axis, 2)
    axis.set_aspect('equal', 'box')
    plt.show()
"""


def bevel_shape_example():
    print(code)
    bevel_shape_linear_interpolation_example()


if __name__ == "__main__":
    bevel_shape_example()