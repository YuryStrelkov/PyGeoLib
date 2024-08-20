from ...common import NUMERICAL_ACCURACY, fast_math, parallel, parallel_range
from ...Numerics.interpolators import bi_linear_interp_pt
from typing import Callable, Tuple, List, Union
from ...Vectors.vector2 import Vector2
import numpy as np


@fast_math
def _interp(_a: float, _b: float, t) -> float:
    return _a + (_b - _a) * t


@fast_math
def _signum(_a: float) -> float:
    return 1.0 if _a >= 0.0 else -1.0


@fast_math
def _squares_nearest_interp(col: int, row: int, dx: float, dy: float):
    return (col + dx * 0.5, row),\
           (col + dx, row + dy * 0.5),\
           (col + dx * 0.5, row + dy),\
           (col, row + dy * 0.5)


@fast_math
def _squares_linear_interp(col: int, row: int, dx: float, dy: float,
                           a_val: float, b_val: float, c_val: float,
                           d_val: float, threshold: float) -> Tuple[Tuple[float, float], ...]:
    d_t = b_val - a_val
    a = (_interp(col, col + dx, _signum(threshold - a_val)), row) if abs(d_t) < NUMERICAL_ACCURACY else \
        (_interp(col, col + dx, (threshold - a_val) / d_t), row)

    d_t = c_val - b_val
    b = (col + dx, _interp(row, row + dy, _signum(threshold - b_val))) if abs(d_t) < NUMERICAL_ACCURACY else\
        (col + dx, _interp(row, row + dy, (threshold - b_val) / d_t))

    d_t = c_val - d_val
    c = (_interp(col, col + dx, _signum(threshold - d_val)), row + dy) if abs(d_t) < NUMERICAL_ACCURACY else \
        (_interp(col, col + dx, (threshold - d_val) / d_t), row + dy)

    d_t = d_val - a_val
    d = (col, _interp(row, row + dy, _signum(threshold - a_val))) if abs(d_t) < NUMERICAL_ACCURACY else \
        (col, _interp(row, row + dy, (threshold - a_val) / d_t))
    return a, b, c, d


@fast_math
def _eval_field_function(field: Callable[[float, float], float],
                         col: int, row: int, dx: float, dy: float) -> Tuple[float, ...]:
    return field(col, row), field(col + dx, row), field(col + dx, row + dy), field(col, row + dy)


@fast_math
def _compute_state(values: Tuple[float, ...], threshold: float) -> int:
    return sum(case if value > threshold else 0 for case, value in zip((8, 4, 2, 1), values))


_SECTIONS_CONNECTION_ALGORYTHM = {
    1: lambda shape, a, b, c, d: shape.append((c, d)),
    2: lambda shape, a, b, c, d: shape.append((b, c)),
    3: lambda shape, a, b, c, d: shape.append((b, d)),
    4: lambda shape, a, b, c, d: shape.append((a, b)),
    5: lambda shape, a, b, c, d: (shape.append((a, d)), shape.append((b, c))),
    6: lambda shape, a, b, c, d: shape.append((a, c)),
    7: lambda shape, a, b, c, d: shape.append((a, d)),
    8: lambda shape, a, b, c, d: shape.append((a, d)),
    9: lambda shape, a, b, c, d: shape.append((a, c)),
    10: lambda shape, a, b, c, d: (shape.append((a, b)), shape.append((c, d))),
    11: lambda shape, a, b, c, d: shape.append((a, b)),
    12: lambda shape, a, b, c, d: shape.append((b, d)),
    13: lambda shape, a, b, c, d: shape.append((b, c)),
    14: lambda shape, a, b, c, d: shape.append((c, d)),
}


def _array_of_points_wrapper(array: np.ndarray, min_bound: Vector2, max_bound: Vector2):
    return lambda x, y: bi_linear_interp_pt((x - min_bound.x) / (max_bound.x -  min_bound.x),
                                            (y - min_bound.y) / (max_bound.y -  min_bound.y),
                                            array, 1.0, 1.0)


@parallel
def _march_squares_2d(field: Callable[[float, float], float],
                      min_bound: Tuple[float, float], max_bound: Tuple[float, float],
                      march_resolution: Tuple[int, int], threshold: float,
                      interpolate: bool) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:

    rows, cols = max(march_resolution[1], 3), max(march_resolution[0], 3)
    cols_ = cols - 1
    rows_ = cols - 1
    dx = (max_bound[0] - min_bound[0]) / cols_
    dy = (max_bound[1] - min_bound[1]) / rows_
    min_bound_y = min_bound[0]
    min_bound_x = min_bound[1]

    shape: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if interpolate:
        for index in parallel_range(cols_ * rows_):
            row_index, col_index = divmod(index, cols_)
            row = row_index * dy + min_bound_y
            col = col_index * dx + min_bound_x
            field_values = _eval_field_function(field, col, row, dx, dy)
            state = _compute_state(field_values, threshold)
            if state in _SECTIONS_CONNECTION_ALGORYTHM:
                a, b, c, d = _squares_linear_interp(col, row, dx, dy, *field_values, threshold)
                _SECTIONS_CONNECTION_ALGORYTHM[state](shape, a, b, c, d)
        return shape

    for index in parallel_range(cols_ * rows_):
        row_index, col_index = divmod(index, cols_)
        row = row_index * dy + min_bound_y
        col = col_index * dx + min_bound_x
        state = _compute_state(_eval_field_function(field, col, row, dx, dy), threshold)
        if state in _SECTIONS_CONNECTION_ALGORYTHM:
            _SECTIONS_CONNECTION_ALGORYTHM[state](shape, *_squares_nearest_interp(col, row, dx, dy))
    return shape


def _dist(a: Vector2, b: Vector2) -> float:
    return (a - b).magnitude


def _find_neighbouring_section(target_line: Union[List[Vector2], Tuple[Vector2, ...]],
                               lines_list: List[List[Vector2]]) -> bool:
    p1, p2 = target_line[0], target_line[-1]
    result = False
    for line in lines_list:
        line_p1, line_p2 = line[0], line[-1]
        if result := (_dist(p1, line_p1) < NUMERICAL_ACCURACY):
            line_iter = iter(target_line)
            next(line_iter)
            # p2 -> ... <- line_p1 -> ... -> line_p2
            while True:
                try:
                    line.insert(0, next(line_iter))
                except StopIteration as _:
                    break
            break

        if result := _dist(p2, line_p1) < NUMERICAL_ACCURACY:
            line_iter = reversed(target_line)
            next(line_iter)
            # p1 -> ... <- line_p1 -> ... -> line_p2
            while True:
                try:
                    line.insert(0, next(line_iter))
                except StopIteration as _:
                    break
            break

        if result := _dist(p1, line_p2) < NUMERICAL_ACCURACY:
            line_iter = iter(target_line)
            next(line_iter)
            # line_p1 -> ... <- line_p2 -> ... -> p2
            while True:
                try:
                    line.append(next(line_iter))
                except StopIteration as _:
                    break
            break

        if result := _dist(p2, line_p2) < NUMERICAL_ACCURACY:
            line_iter = reversed(target_line)
            next(line_iter)
            # line_p1 -> ... <- line_p2 -> ... -> p1
            while True:
                try:
                    line.append(next(line_iter))
                except StopIteration as _:
                    break
            break
    if not result:
        lines_list.append([*target_line])
    return result


def _connect_sects(sects: List[Tuple[Vector2, Vector2]]) -> List[List[Vector2]]:
    if len(sects) == 0:
        return []
    lines = [[*sects.pop()]]
    # simple bounded sections connect
    for i in range(3):
        while len(sects) != 0:
            sect = sects.pop()
            _find_neighbouring_section(sect, lines)
        if i != 2:
            lines, sects = [lines.pop()], lines
            continue
        break
    return lines


def _clear_line(line: List[Vector2], angle_threshold: float = 0.125, distance_threshold: float = 0.21) -> List[Vector2]:
    clean_list = [line[0]]
    for pl, pc, pr in zip(line[:-2], line[1:-1], line[2:]):
        if abs(Vector2.cross((pl - pc).normalize(), (pr - pc).normalize())) > angle_threshold:
            clean_list.append(pc)
        elif Vector2.distance(pc, clean_list[-1]) > distance_threshold:
            clean_list.append(pc)
    clean_list.append(line[-1])
    return clean_list


def squares_marching_2d(field: Union[Callable[[float, float], float], np.ndarray],
                        min_bound: Vector2 = None,
                        max_bound: Vector2 = None,
                        march_resolution: Tuple[int, int] = None,
                        threshold: float = 0.5,
                        interpolate: bool = True) -> List[List[Vector2]]:

    march_resolution = (128, 128) if march_resolution is None else march_resolution
    assert len(march_resolution) == 2 and all(isinstance(v, int) for v in march_resolution)

    min_bound = Vector2(-5.0, -5.0) if min_bound is None else min_bound
    assert isinstance(min_bound, Vector2)

    max_bound = Vector2( 5.0,  5.0) if max_bound is None else max_bound
    assert isinstance(max_bound, Vector2)

    if isinstance(field, np.ndarray):
        field[:, 0] = 1.0e32
        field[:, -1] = 1.0e32
        field[0, :] = 1.0e32
        field[-1, :] = 1.0e32
        field = _array_of_points_wrapper(field, min_bound, max_bound)
        shape = _march_squares_2d(field, tuple(min_bound), tuple(max_bound), march_resolution, threshold, interpolate)
        shape = list((Vector2(*p1), Vector2(*p2)) for p1, p2 in shape)
        return [_clear_line(line) for line in _connect_sects(shape)]
    if isinstance(field, Callable):
        shape = _march_squares_2d(field, tuple(min_bound), tuple(max_bound), march_resolution, threshold, interpolate)
        shape = list((Vector2(*p1), Vector2(*p2)) for p1, p2 in shape)
        return [_clear_line(line) for line in _connect_sects(shape)]
    raise ValueError(f"unsupported field type: {type(field)}")
