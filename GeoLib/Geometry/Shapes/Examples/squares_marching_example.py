from ..Curves import squares_marching_2d
from matplotlib import pyplot as plt
from math import cos, sin, sqrt
from ...Vectors import Vector2
from ...common import PI
import numpy as np


def squares_marching_example():
    scale = 100
    t = np.linspace(0.0, PI, 128)
    field = np.array(tuple(tuple(cos(sqrt(scale * x * x * x + scale * (y + 1.0) * y**2)) for y in t) for x in t))
    sections = squares_marching_2d(field, Vector2(0.0, 0.0), Vector2(t[-1], t[-1]),
                                   (t.size, t.size), threshold=0.5, interpolate=True)

    for index, points in enumerate(sections):
        plt.plot(tuple(p.x for p in points), tuple(p.y for p in points))
    plt.gca().axis('equal')
    plt.show()
