from ..Transformations.transform_2d import Transform2d
from typing import List, Iterable, Dict, Any
from ..Vectors.vector2 import Vector2
from .ray_tracing_common import *
import matplotlib.pyplot as plt
import logging
import math

"""
#######################################################################################################################
#################################                   RAY TRACING 2 D                   #################################
#######################################################################################################################
"""


"""
Implements raytracing using simple spherical surface.
"""


def intersect_sphere_2d(direction: Vector2, origin: Vector2, radius: float) -> float:
    dr = Vector2(0, -radius) - origin
    dre = Vector2.dot(dr, direction)
    det = dre ** 2 - Vector2.dot(dr, dr) + radius * radius
    if det < 0:
        return -1.0
    det = math.sqrt(det)
    t1, t2 = dre + det, dre - det
    if t1 < 0 and t2 < 0:
        return -1.0
    if t1 * t2 < 0:
        return max(t1, t2)
    return t2


def intersect_flat_surface_2d(direction: Vector2, origin: Vector2, normal: Vector2) -> float:
    rn = Vector2.dot(origin, -normal)
    return rn * (1.0 / Vector2.dot(direction, normal))


def _trace_surface_2d(direction: Vector2, origin: Vector2, radius: float) -> Tuple[float, Vector2, Vector2]:
    if abs(radius) <= NUMERICAL_ACCURACY:
        normal = Vector2(0.0, 1.0 if radius >= 0 else -1.0)
        t = intersect_flat_surface_2d(direction, origin, normal)
        return t,  direction * t + origin, normal
    t = intersect_sphere_2d(direction, origin, radius)
    ray_end = direction * t + origin
    return t, ray_end, Vector2(-ray_end.x, -ray_end.y - radius).normalized


def trace_surface_2d(rd: Vector2, ro: Vector2, radius: float, transform: Transform2d = None) -> \
        Tuple[float, Vector2, Vector2]:
    """
    Рассчитывает пересечение луча и поверхности
    @param rd: направление луча (единичный вектор)
    @param ro: координата начала луча
    @param radius: радиус поверхности
    @param transform: пространственная трансформация поверхности
    @return: длина луча от его начала до точки пересечения с поверхностью, координату точки пересечения с поверхностью
    нормаль поверхности в точке пересечения.
    """
    if not transform:
        return _trace_surface_2d(rd, ro, radius)
    _rd = transform.inv_transform_vect(rd, 0.0)
    _ro = transform.inv_transform_vect(ro, 1.0)
    t, re, rn = _trace_surface_2d(_rd, _ro, radius)
    return (0.0, ro, re) if t < 0 else (t, transform.transform_vect(re), transform.transform_vect(rn, 0.0))


def reflect_2d(rd: Vector2, ro: Vector2, radius: float, transform: Transform2d = None) -> \
        Tuple[float, Vector2, Vector2]:
    """
    Рассчитывает отражение луча от поверхности
    @param rd: направление луча (единичный вектор)
    @param ro: координата начала луча
    @param radius: радиус поверхности
    @param transform: пространственная трансформация поверхности
    @return: длина луча от его начала до точки пересечения с поверхностью, координату точки пересечения с поверхностью
    нормаль поверхности в точке пересечения.
    """
    t, re, rn = trace_surface_2d(rd, ro, radius, transform)
    return (0.0, ro, re) if t < 0 else (t, re, Vector2.reflect(rd, rn))


def refract_2d(rd: Vector2, ro: Vector2, radius: float, ri1: float, ri2: float, transform: Transform2d = None) -> \
        Tuple[float, Vector2, Vector2]:
    """
    Рассчитывает преломление луча поверхностью
    @param rd: направление луча (единичный вектор)
    @param ro: координата начала луча
    @param radius: радиус поверхности
    @param ri1: коэффициент преломления перед поверхностью
    @param ri2: коэффициент преломления после поверхности
    @param transform: пространственная трансформация поверхности
    @return: длина луча от его начала до точки пересечения с поверхностью, координату точки пересечения с поверхностью
    нормаль поверхности в точке пересечения.
    """
    t, re, rn = trace_surface_2d(rd, ro, radius, transform)
    return (0.0, ro, re) if t < 0 else (t, re, Vector2.refract(rd, rn, ri1, ri2))


def build_shape_2d(radius: float, semi_diam: float,
                   transform: Transform2d = None, steps: int = 16) -> Tuple[List[float], List[float]]:
    if abs(semi_diam) <= NUMERICAL_ACCURACY:
        return [], []
    if abs(radius) <= NUMERICAL_ACCURACY:
        xs, ys = [-semi_diam, semi_diam], [0, 0]
    else:
        da = 2 * semi_diam / (steps - 1)
        xs = [da * i - semi_diam for i in range(steps)]
        sgn = -1.0 if radius > 0 else 1.0
        ys = [-radius - sgn * math.sqrt((radius * radius - x * x)) for x in xs]
    if not transform:
        return xs, ys
    for index, (xi, yi) in enumerate(zip(xs, ys)):
        (xt, yt) = transform.transform_vect(Vector2(xi, yi), 1.0)
        xs[index] = xt
        ys[index] = yt
    return xs, ys


def lens_shape_2d(r1: float, r2: float, s_dia1: float, s_dia2: float,
                  transform_1: Transform2d = None, transform_2: Transform2d = None, steps: int = 16) -> \
        Tuple[List[float], List[float]]:
    s_dia = max(s_dia1, s_dia2)
    xs, ys = build_shape_2d(r1, s_dia, transform_1, steps)
    xs1, ys1 = build_shape_2d(r2, s_dia, transform_2, steps)
    xs1.reverse()
    ys1.reverse()
    xs.extend(xs1)
    ys.extend(ys1)
    xs.append(xs[0])
    ys.append(ys[0])
    return xs, ys


def trace_ray_2d(rd: Vector2, ro: Vector2,  # начало и направление луча
                 surfaces_r: Iterable[float],  # список поверхностей (только радиусы)
                 surfaces_t: Iterable[Transform2d],  # список трансформаций поверхностей
                 surfaces_p: Iterable[Dict[str, Any]]) -> \
        Tuple[List[Vector2], List[Vector2]]:  # дополнительные параметры поверхностей
    """
    Делает трассировку луча через набор сферических поверхностей
    @param ro: начало луча
    @param rd: направление луча
    @param surfaces_r: список поверхностей (только радиусы)
    @param surfaces_t: список трансформаций поверхностей
    @param surfaces_p: дополнительные параметры поверхностей, которые хранятся в виде словаря, например
    {'material': 'mirror'} - для зеркала
    или {'material': 'glass', 'glass-params': (1.333, 1.0)} для преломляющей поверхности.
    @return: список точек пересечения с поверхностями и список направления лучей в точках пересечения
    """
    points = [ro]
    directions = [rd]
    for surface_index, (s_r, s_t, s_p) in enumerate(zip(surfaces_r, surfaces_t, surfaces_p)):
        if MATERIAL not in s_p:
            continue
        if s_p[MATERIAL] == MIRROR:
            t, _re, _rd = reflect_2d(directions[-1], points[-1], s_r, s_t)
            if t < 0:
                break
            points.append(_re)
            directions.append(_rd)
        if s_p[MATERIAL] == IMAGE_OBJECT:
            t, _re, _rd = trace_surface_2d(directions[-1], points[-1], s_r, s_t)
            if t < 0:
                break
            points.append(_re)
            directions.append(_rd)
        if s_p[MATERIAL] == GLASS:
            if GLASS_PARAMS not in s_p:
                continue
            ri1, ri2 = s_p[GLASS_PARAMS]
            try:
                t, _re, _rd = refract_2d(directions[-1], points[-1], s_r, ri1, ri2, s_t)
            except ValueError as error:
                logging.warning(f"|\ttrace-error: error occurs at surface №{surface_index}, surface will be ignored.\n"
                                f"|\terror-info : {error}")
                continue
            except ZeroDivisionError as error:
                logging.warning(f"|\ttrace-error: error occurs at surface №{surface_index}, surface will be ignored.\n"
                                f"|\terror-info : {error}")
                continue
            if t < 0:
                break
            points.append(_re)
            directions.append(_rd)
    return points, directions


def draw_scheme_2d(surfaces_r: Iterable[float],  # список поверхностей (только радиусы)
                   aperture_r: Iterable[float],
                   surfaces_t: Iterable[Transform2d],  # список трансформаций поверхностей
                   surfaces_p: Iterable[Dict[str, Any]], axis=None, steps: int = 32):
    axis = axis if axis else plt.gca()
    iter_surfaces_r = iter(surfaces_r)
    iter_aperture_r = iter(aperture_r)
    iter_surfaces_t = iter(surfaces_t)
    iter_surfaces_p = iter(surfaces_p)
    surf_index = -1
    while True:
        surf_index += 1
        try:
            r1 = next(iter_surfaces_r)
            a1 = next(iter_aperture_r)
            t1 = next(iter_surfaces_t)
            p1 = next(iter_surfaces_p)
            if MATERIAL not in p1:
                x, y = build_shape_2d(r1, a1, t1, steps)
                axis.plot(x, y, ':k')
                continue
            if p1[MATERIAL] == IMAGE_OBJECT:
                x, y = build_shape_2d(r1, a1, t1, steps)
                axis.plot(x, y, 'g', linewidth=1.5)
            if p1[MATERIAL] == SOURCE_OBJECT:
                x, y = build_shape_2d(r1, a1, t1, steps)
                axis.plot(x, y, 'r', linewidth=1.5)
            if p1[MATERIAL] == DUMMY_OBJECT:
                x, y = build_shape_2d(r1, a1, t1, steps)
                axis.plot(x, y, '--k', linewidth=0.75)
            if p1[MATERIAL] == MIRROR:
                x, y = build_shape_2d(r1, a1, t1, steps)
                axis.plot(x, y, 'k')
                continue
            if p1[MATERIAL] != GLASS:
                continue
            r2 = next(iter_surfaces_r)
            a2 = next(iter_aperture_r)
            t2 = next(iter_surfaces_t)
            _  = next(iter_surfaces_p)
            x, y = lens_shape_2d(r1, r2, a1, a2, t1, t2)
            axis.plot(x, y, 'b')
        except ValueError as error:
            logging.warning(f"\tshape-error : error while building surface {surf_index}, surface will not be drawn...\n"
                            f"\terror-info  : {error}")
            continue
        except StopIteration:
            logging.info(f"\tdraw-info   : file: drawing successfully done...\n")
            break
    axis.set_aspect('equal', 'box')
    axis.set_xlabel("z, [mm]")
    axis.set_ylabel("x, [mm]")
    return axis


def tracing_2d_test():
    surfaces_r = [1e12, -350, -350, 1e12, -350, 350, 550, 350, 1e12]  # : Iterable[float]
    aperture_r = [50, 50, 50, 55, 50, 50, 50, 20, 20]  # : Iterable[float]
    surfaces_t = [Transform2d(pos=Vector2(-50,  0.0), angle=90.0),
                  Transform2d(pos=Vector2(-15,  0.0), angle=90.0),
                  Transform2d(pos=Vector2(-5,   0.0), angle=90.0),
                  Transform2d(pos=Vector2(0,    0.0), angle=90.0),
                  Transform2d(pos=Vector2(0,    0.0), angle=90.0),
                  Transform2d(pos=Vector2(30,   0.0), angle=90.0),
                  Transform2d(pos=Vector2(125,  0.0), angle=90.0),
                  Transform2d(pos=Vector2(30.1, 0.0), angle=90.0),
                  Transform2d(pos=Vector2(400,  0.0), angle=90.0)]  # : Iterable[Transform2d]
    surfaces_p = [{MATERIAL: SOURCE_OBJECT},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.66)},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.66, 1.0)},
                  {MATERIAL: DUMMY_OBJECT},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.333)},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.333, 1.0)},
                  {MATERIAL: MIRROR},
                  {MATERIAL: MIRROR},
                  {MATERIAL: IMAGE_OBJECT}]  # : Iterable[Dict[str, Any]]
    for i in range(-10, 10):
        positions, directions = trace_ray_2d(Vector2(1, 0), Vector2(-50, i * 3), surfaces_r, surfaces_t, surfaces_p)
        xs = [v.x for v in positions]
        ys = [v.y for v in positions]
        plt.plot(xs, ys, 'r')
    draw_scheme_2d(surfaces_r, aperture_r, surfaces_t, surfaces_p, plt.gca())
    plt.show()
