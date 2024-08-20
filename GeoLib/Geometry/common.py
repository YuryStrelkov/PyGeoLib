from typing import Tuple
import inspect
import sys

NUMERICAL_FORMAT_4F = ">10.4f"
NUMERICAL_FORMAT_8F = ">10.8f"
NUMERICAL_MIN_VALUE = -1e32
NUMERICAL_MAX_VALUE = 1e32
NUMERICAL_ACCURACY = 1e-9
PI = 3.141592653589793
TWO_PI = 2.0 * PI
HALF_PI = 0.5 * PI
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI


_indent = ''


def set_indent_level(level: int):
    global _indent
    _indent = '\t' * level


def indent():
    return _indent


def _numba_error_decorator(func):
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('Exception in %s : %s' % (func.__name__, e))
    return wrapped


class CustomDecorator:

    def __init__(self, f):
        self.f = f

    def __call__(self, cls):
        for name, m in inspect.getmembers(cls, inspect.ismethod):
            setattr(cls, name, self.f(m))
        return cls


parallel_range = None
fast_math      = None
parallel       = None


try:
    import numba
    parallel_range = numba.prange
    fast_math      = CustomDecorator(numba.njit(fastmath=True))
    parallel       = CustomDecorator(numba.njit(parallel=True, fastmath=True))
except ImportError as err:
    parallel_range = range
    fast_math      = CustomDecorator(_numba_error_decorator)
    parallel       = CustomDecorator(_numba_error_decorator)
    print(f"ImportError:: {err}")


def assert_version(version_major: int, version_minor: int) -> bool:
    """
    Сравнение необходимой версии интерпретатора Python с текущей
    :param version_major:
    :param version_minor:
    :return:
    """
    if sys.version_info.major < version_major:
        return False
    if sys.version_info.minor < version_minor:
        return False
    return True


def pyton_version() -> Tuple[int, int]:
    """
    Текущая версия интерпретатора Python
    @return: [Мажорная версия: int, Минорная версия: int]
    """
    return sys.version_info.major, sys.version_info.minor


DATA_CLASS_INSTANCE_ARGS = {'slots': True} if assert_version(3, 10) else {}
