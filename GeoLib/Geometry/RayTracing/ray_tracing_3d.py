from typing import Iterable, List, Any, Dict
from .ray_tracing_common import *
from .rays_source import rect_source
import matplotlib.pyplot as plt
import numpy as np
import logging
import math

"""
#######################################################################################################################
#################################                   RAY TRACING 3 D                   #################################
#######################################################################################################################
"""


"""
Implements raytracing using simple spherical surface.
"""


def intersect_sphere_3d(direction: Vector3, origin: Vector3, radius: float) -> float:
    dr = Vector3(0, 0, -radius) - origin
    dre = Vector3.dot(dr, direction)
    det = dre ** 2 - Vector3.dot(dr, dr) + radius * radius
    if det < 0:
        return -1.0
    det = math.sqrt(det)
    t1, t2 = dre + det, dre - det
    if t1 < 0 and t2 < 0:
        return -1.0
    if t1 * t2 < 0:
        return max(t1, t2)
    return t2


def intersect_flat_surface_3d(direction: Vector3, origin: Vector3, normal: Vector3) -> float:
    rn = Vector3.dot(origin, -normal)
    return rn * (1.0 / Vector3.dot(direction, normal))


def _trace_surface_3d(direction: Vector3, origin: Vector3, radius: float) -> Tuple[float, Vector3, Vector3]:
    if abs(radius) <= NUMERICAL_ACCURACY:
        normal = Vector3(0.0, 0.0, 1.0 if radius >= 0 else -1.0)
        t = intersect_flat_surface_3d(direction, origin, normal)
        return t, direction * t + origin, normal
    t = intersect_sphere_3d(direction, origin, radius)
    ray_end = direction * t + origin
    return t, ray_end, Vector3(-ray_end.x, -ray_end.y, -ray_end.z - radius).normalized


def trace_surface_3d(rd: Vector3, ro: Vector3, radius: float, transform: Transform3d = None) -> \
        Tuple[float, Vector3, Vector3]:
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
        return _trace_surface_3d(rd, ro, radius)
    _rd = transform.inv_transform_vect(rd, 0.0)
    _ro = transform.inv_transform_vect(ro, 1.0)
    t, re, rn = _trace_surface_3d(_rd, _ro, radius)
    return (0.0, ro, re) if t < 0 else (t, transform.transform_vect(re), transform.transform_vect(rn, 0.0))


def reflect_3d(rd: Vector3, ro: Vector3, radius: float, transform: Transform3d = None) -> Tuple[float, Vector3, Vector3]:
    """
    Рассчитывает отражение луча от поверхности
    @param rd: направление луча (единичный вектор)
    @param ro: координата начала луча
    @param radius: радиус поверхности
    @param transform: пространственная трансформация поверхности
    @return: длина луча от его начала до точки пересечения с поверхностью, координату точки пересечения с поверхностью
    нормаль поверхности в точке пересечения.
    """
    t, re, rn = trace_surface_3d(rd, ro, radius, transform)
    return (0.0, ro, re) if t < 0 else (t, re, Vector3.reflect(rd, rn))


def refract_3d(rd: Vector3, ro: Vector3, radius: float, ri1: float, ri2: float, transform: Transform3d = None) -> \
        Tuple[float, Vector3, Vector3]:
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
    t, re, rn = trace_surface_3d(rd, ro, radius, transform)
    return (0.0, ro, re) if t < 0 else (t, re, Vector3.refract(rd, rn, ri1, ri2))


def trace_ray_3d(rd: Vector3, ro: Vector3,  # начало и направление луча
                 surfaces_r: Iterable[float],  # список поверхностей (только радиусы)
                 surfaces_t: Iterable[Transform3d],  # список трансформаций поверхностей
                 surfaces_p: Iterable[Dict[str, Any]]) -> \
        Tuple[List[Vector3], List[Vector3]]:  # дополнительные параметры поверхностей
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
        try:
            if s_p[MATERIAL] == MIRROR:
                t, _re, _rd = reflect_3d(directions[-1], points[-1], s_r, s_t)
                if t < 0:
                    break
                points.append(_re)
                directions.append(_rd)
            if s_p[MATERIAL] == IMAGE_OBJECT:
                t, _re, _rd = trace_surface_3d(directions[-1], points[-1], s_r, s_t)
                if t < 0:
                    break
                points.append(_re)
                directions.append(_rd)
            if s_p[MATERIAL] == GLASS:
                if GLASS_PARAMS not in s_p:
                    continue
                ri1, ri2 = s_p[GLASS_PARAMS]
                t, _re, _rd = refract_3d(directions[-1], points[-1], s_r, ri1, ri2, s_t)
                if t < 0:
                    break
                points.append(_re)
                directions.append(_rd)
        except ValueError as error:
            logging.warning(f"|\ttrace-error: error occurs at surface №{surface_index}, surface will be ignored.\n"
                            f"|\terror-info : {error}")
            break
        except ZeroDivisionError as error:
            logging.warning(f"|\ttrace-error: error occurs at surface №{surface_index}, surface will be ignored.\n"
                            f"|\terror-info : {error}")
            break
    return points, directions


def _edge_thickness(radius, px, py):
    return -radius - (-1.0 if radius > 0 else 1.0) * math.sqrt((max(radius * radius - px * px - py * py, 0.0)))


def lens_shape_3d(r1: float, r2: float, s_dia1: float, s_dia2: float,
                  transform_1: Transform3d = None, transform_2: Transform3d = None, steps: int = 16) -> \
        Dict[str, Tuple[np.ndarray, ...]]:
    s_dia = max(s_dia1, s_dia2)
    d_pos = (transform_2.origin - transform_1.origin).magnitude
    s_thick_1 = _edge_thickness(r1, 0, s_dia1)
    s_thick_2 = _edge_thickness(r2, 0, s_dia2)
    xs0, ys0, zs0 = build_shape_3d(r1, s_dia, transform_1, steps)
    xs1, ys1, yz1 = build_shape_3d(r2, s_dia, transform_2, steps)
    xs2, ys2, yz2 = build_cylinder_shape_3d(s_dia1, s_dia2, s_thick_1,
                                            d_pos - s_thick_1 + s_thick_2, transform_1, 4 * steps)
    return {'front-surf': (xs0, ys0, zs0), 'side-surf': (xs1, ys1, yz1), 'back-surf': (xs2, ys2, yz2)}


def draw_scheme_3d(surfaces_r: Iterable[float],  # список поверхностей (только радиусы)
                   aperture_r: Iterable[float],
                   surfaces_t: Iterable[Transform3d],  # список трансформаций поверхностей
                   surfaces_p: Iterable[Dict[str, Any]], axis=None, steps: int = 63):
    axis = axis if axis else plt.axes(projection='3d')
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
                x, y, z = build_shape_3d(r1, a1, t1, steps)
                axis.contour3D(x, y, z, antialiased=False, color='grey')
                continue
            if p1[MATERIAL] == IMAGE_OBJECT:
                x, y, z = build_shape_3d(r1, a1, t1, steps)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='green')
            if p1[MATERIAL] == SOURCE_OBJECT:
                x, y, z  = build_shape_3d(r1, a1, t1, steps)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='red')
            if p1[MATERIAL] == DUMMY_OBJECT:
                x, y, z = build_shape_3d(r1, a1, t1, steps)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='grey')
            if p1[MATERIAL] == MIRROR:
                x, y, z  = build_shape_3d(r1, a1, t1, steps)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='white')
                continue
            if p1[MATERIAL] != GLASS:
                continue
            r2 = next(iter_surfaces_r)
            a2 = next(iter_aperture_r)
            t2 = next(iter_surfaces_t)
            _  = next(iter_surfaces_p)
            surfaces = lens_shape_3d(r1, r2, a1, a2, t1, t2)
            for surf in surfaces.values():
                axis.plot_surface(*surf, linewidths=0.0, antialiased=True, color='blue', edgecolor="none")
        except ValueError as error:
            logging.warning(f"\tshape-error : error while building surface {surf_index}, surface will not be drawn...\n"
                            f"\terror-info  : {error}")
            continue
        except StopIteration:
            logging.info(f"\tdraw-info   : file: drawing successfully done...\n")
            break
    axis.set_aspect('equal', 'box')
    axis.set_xlabel("x, [mm]")
    axis.set_ylabel("y, [mm]")
    axis.set_zlabel("z, [mm]")
    return axis


def tracing_3d_test():
    surfaces_r = [1e12, -350, -350, 1e12, -350, 350, 550, 350, 1e12]  # : Iterable[float]
    aperture_r = [50, 50, 50, 55, 50, 50, 50, 20, 20]  # : Iterable[float]
    surfaces_t = [Transform3d(pos=Vector3(70 + -50,  0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + -15,  0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + -5,   0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + 0,    0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + 0,    0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + 30,   0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + 125,  0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + 30.1, 0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0)),
                  Transform3d(pos=Vector3(70 + 150,  0.0, 0.0), angles=Vector3(0.0, 90.0, 0.0))]
    # : Iterable[Transform2d]
    surfaces_p = [{MATERIAL: SOURCE_OBJECT},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.66)},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.66, 1.0)},
                  {MATERIAL: DUMMY_OBJECT},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.333)},
                  {MATERIAL: GLASS, GLASS_PARAMS: (1.333, 1.0)},
                  {MATERIAL: MIRROR},
                  {MATERIAL: MIRROR},
                  {MATERIAL: IMAGE_OBJECT}]
    # : Iterable[Dict[str, Any]]
    axis = plt.axes(projection='3d')
    for rd, ro in rect_source():
        positions, directions = trace_ray_3d(rd, ro, surfaces_r, surfaces_t, surfaces_p)
        xs = [v.x for v in positions]
        ys = [v.y for v in positions]
        zs = [v.z for v in positions]
        plt.plot(xs, ys, zs, 'r')
    draw_scheme_3d(surfaces_r, aperture_r, surfaces_t, surfaces_p, axis)
    plt.show()
