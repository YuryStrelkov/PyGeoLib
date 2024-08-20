"""
Base geometry primitives library
"""
__version__ = '0.1.25'
__license__ = "GNU Lesser General Public License v3"
from .common import DATA_CLASS_INSTANCE_ARGS, assert_version, pyton_version
from .common import NUMERICAL_MAX_VALUE, NUMERICAL_MIN_VALUE, PI, TWO_PI, HALF_PI
from .common import NUMERICAL_ACCURACY, NUMERICAL_FORMAT_4F, NUMERICAL_FORMAT_8F
from .common import DEG_TO_RAD, RAD_TO_DEG, parallel_range, fast_math, parallel, indent, set_indent_level
from .Vectors import *
from .Matrices import *
from .Shapes import *
from .Transformations import *
from .Bounds import *
from .RayTracing import *
from .camera import Camera
from .Numerics import *
from .Meshes import *
from .color import Color, create_color_rgb, create_color_rgba