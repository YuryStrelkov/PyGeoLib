from .ray_tracing_common import NUMERICAL_ACCURACY, MATERIAL, SOURCE_OBJECT, DUMMY_OBJECT
from .ray_tracing_common import IMAGE_OBJECT, MIRROR, GLASS, GLASS_PARAMS
from ..Transformations.transform_3d import Transform3d
from typing import Tuple, List, Iterable, Dict, Union
from ..Vectors.vector2 import Vector2
from ..Vectors.vector3 import Vector3
from ..Matrices.matrix3 import Matrix3
from matplotlib import pyplot as plt
from .rays_source import rect_source
import numpy as np
import logging
import math


def _surf_intersect(rd: Vector3, ro: Vector3, m_surf: Matrix3,
                    b_surf: Vector3, k_surf: float, orientation: float) -> float:
    """
    xAx + (b,x) + k = 0
    x = y - d
    x = r0 + et - d
    (r0 + et - d)A(r0 + et - d) + (b, r0 + et - d) + k = 0
    (r0 + et - d)(Ar0 + Aet - Ad) + (b,r0) + (b,e) t - (b,d) + k = 0
    r0Ar0 + eAr0t - dAr0 + r0Aet + eAet^2 - dAet - r0Ad - eAdt + dAd + (b,r0) + (b,e) t - (b,d) + k = 0
    eAet^2 + (eAr0 + r0Ae)t  - (dAe + eAd)t + (b,e) t + r0Ar0 + dAd - r0Ad + (b,r0) - dAr0 - (b,d) + k = 0
    a = eAe
    b = eAr0 + r0Ae - dAe - eAd + (b,e) = 2 eAr0 - 2 dAe + (b,e)
    c = (r0 - d, Ar0) + (d - r0, Ad) + (b, r0) - (b, d) + k
    c = (r0 - d, Ar0) - (r0 - d, Ad) + (b, r0 - d)  + k
    c = (r0 - d, Ar0) - (r0 - d, Ad) - (r0 - d, b)  + k
    @param ro:
    @param rd:
    @param m_surf:
    @param b_surf:
    @param k_surf:
    @return:
    """
    ar = m_surf * ro  # A * r0
    ae = m_surf * rd  # A * e0
    # a = eAe
    a = Vector3.dot(rd, ae)  # eAe
    # b = 2 eAr0 - 2 dAe + (b,e)
    b = Vector3.dot(ro, ae) + Vector3.dot(rd, ar) + Vector3.dot(rd, b_surf)
    #  c = (r0 - d, Ar0) - (r0 - d, Ad)  - (r0 - d, b)  + k
    c = Vector3.dot(ro, ar) + Vector3.dot(ro, b_surf) + k_surf
    if abs(a) < NUMERICAL_ACCURACY:
        return -c / b
    det = b * b - 4 * a * c
    if det < 0:
        return -1.0
    if det < 0:
        return -1.0
    det = math.sqrt(det)
    a = 1.0 / (2.0 * a)
    t1, t2 = (-b + det) * a, (-b - det) * a
    if t1 < 0 and t2 < 0:
        return -1.0
    if t1 * t2 < 0:
        return max(t1, t2)
    return max(t1, t2) if orientation > 1.0 else min(t1, t2)


def _surf_sag(x: float, y: float, m_surf: Matrix3, b_surf: Vector3, k_surf: float, s_orientation: float) -> Vector3:
    a = m_surf.m22
    b = b_surf.z + 2 * x * m_surf.m02 + 2 * y * m_surf.m12
    c = x * x * m_surf.m00 + 2 * x * y * m_surf.m01 + y * y * m_surf.m11 + x * b_surf.x + y * b_surf.y + k_surf
    if abs(a) < NUMERICAL_ACCURACY:
        return Vector3(x, y, -c / b) if abs(b) > NUMERICAL_ACCURACY else Vector3(x, y, 0.0)
    det = b * b - 4 * a * c
    if det < 0:
        return Vector3(x, y, 0.0)
    det = math.sqrt(det)
    t1, t2 = (-b + det) / (2 * a), (-b - det) / (2 * a)
    if t1 < 0 and t2 < 0:
        return Vector3(x, y, 0.0)
    if t1 * t2 < 0:
        return Vector3(x, y, (max(t1, t2) if s_orientation < 0.0 else min(t1, t2)))
    return Vector3(x, y, t2)


def _trace_surface(rd: Vector3, ro: Vector3, m_surf: Matrix3,
                   b_surf: Vector3, k_surf: float, orientation: float) -> Tuple[float, Vector3, Vector3]:
    t = _surf_intersect(rd, ro, m_surf, b_surf, k_surf, orientation)
    ray_end = rd * t + ro
    return t, ray_end, (2 * (m_surf * ray_end) + b_surf).normalized


SPHERICAL_SURFACE = 0
ELLIPSOIDAL_SURFACE = 1
HYPERBOLIC_SURFACE = 2
PARABOLOID_SURFACE = 3
ELLIPTIC_PARABOLOID_SURFACE = 4
HYPERBOLIC_PARABOLOID_SURFACE = 5
CONIC_SURFACE = 6
USER_DEFINED_SURFACE = 7

SURF_TYPES_STR = {
    SPHERICAL_SURFACE: 'SPHERICAL_SURFACE',
    ELLIPSOIDAL_SURFACE: 'ELLIPSOIDAL_SURFACE',
    HYPERBOLIC_SURFACE: 'HYPERBOLIC_SURFACE',
    PARABOLOID_SURFACE: 'PARABOLOID_SURFACE',
    ELLIPTIC_PARABOLOID_SURFACE: 'ELLIPTIC_PARABOLOID_SURFACE',
    HYPERBOLIC_PARABOLOID_SURFACE: 'HYPERBOLIC_PARABOLOID_SURFACE',
    CONIC_SURFACE: 'CONIC_SURFACE',
    USER_DEFINED_SURFACE: 'USER_DEFINED_SURFACE'
}
NO_MATERIAL = '\"-\"'


class OpticalSurface:
    """
    Implements second order surface with commonly known formulae: (x,A,x) + (b,x) + c = 0.
    Where A - 3x3 symmetric matrix; x - tree dimensional vector of space coordinates; b - tree dimensional vector;
    c - scalar value.
    """
    __slots__ = ('_transform', '_m_param', '_a_param', '_b_param', '_d_param',
                 '_direction', '_k_param', '_s_type', '_material')

    def __init__(self):
        # default is sphere with radius of value 10 and positive curvature
        surf_radius = 10
        self._transform = Transform3d()
        self._m_param = Matrix3.identity() / (surf_radius * surf_radius)
        self._a_param = Vector2(0, surf_radius * 0.5)
        self._b_param = Vector3(0, 0, 0)
        self._d_param = Vector3(0, 0, -surf_radius)
        self._k_param = -1.0
        self._s_type = SPHERICAL_SURFACE
        self._direction = 1.0 if surf_radius > 0.0 else -1.0
        self._material = {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.66)}

    @property
    def surface_orientation(self) -> float:
        return self._direction

    def __str__(self):
        return f"{{\n" \
               f"\"surf_type\": \"{SURF_TYPES_STR[self.surf_type]}\",\n" \
               f"\"surf_mat\" :\n{self.surface_matrix},\n" \
               f"\"param_b\"  : {self.b_params},\n" \
               f"\"param_d\"  : {self.d_params},\n" \
               f"\"param_k\"  : {self.k_param},\n" \
               f"\"material\" : {self._material},\n" \
               f"\"transform\":\n{self._transform}\n" \
               f"}}"

    @property
    def aperture(self) -> Vector2:
        return self._a_param

    @aperture.setter
    def aperture(self, value: Union[Vector2, float]) -> None:
        if isinstance(value, float) or isinstance(value, int):
            self._a_param.y = max(self._a_param.y, value)
            self._a_param.x = min(self._a_param.x, self._a_param.y)
            return
        if isinstance(value, Vector2):
            self._a_param = Vector2(min(*value), max(*value))
            return
        raise ValueError(f"incorrect surface aperture assignment\n error value: {value}")

    @property
    def aperture_min(self) -> float:
        return self._a_param.x

    @property
    def aperture_max(self) -> float:
        return self._a_param.y

    @property
    def surf_type(self) -> int:
        return self._s_type

    @property
    def surface_matrix(self) -> Matrix3:
        return self._m_param

    @property
    def b_params(self) -> Vector3:
        return self._b_param

    @property
    def d_params(self) -> Vector3:
        return self._d_param

    @property
    def k_param(self) -> float:
        return self._k_param

    @property
    def transform(self) -> Transform3d:
        return self._transform

    @property
    def material(self) -> dict:
        return self._material

    def intersect_ray(self, direction: Vector3, origin: Vector3) -> Tuple[float, Vector3, Vector3]:
        """
        Рассчитывает расстояние до точки пересечения, координаты точки пересечения луча и нормаль в этой точке
        @param direction: направление луча
        @param origin: начало луча
        @return: длина луча до точки пересечения, координата точки пересечения, нормаль в точке пересечения
        """
        _rd = self.transform.inv_transform_vect(direction, 0.0)
        _ro = self.transform.inv_transform_vect(origin, 1.0) + self.d_params
        t, re, rn = _trace_surface(_rd, _ro, self.surface_matrix, self.b_params, self.k_param, self.surface_orientation)
        return (0.0, origin, direction) if t < 0 else \
            (t, self.transform.transform_vect(re - self.d_params, 1.0), self.transform.transform_vect(rn, 0.0))

    def surf_sag(self, x: float, y: float) -> Vector3:
        sag = _surf_sag(x, y, self.surface_matrix, self.b_params, self.k_param, self.surface_orientation)
        if abs(self.d_params.z) < 1e6:
            sag -= self.d_params
        return self.transform.transform_vect(sag, 1.0)

    def surf_shape(self, steps_r: int = 32, steps_angle: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = np.linspace(0, np.pi * 2.0, steps_angle)
        radius = self.aperture_min + np.sqrt(np.linspace(0.0, 1.0, steps_r)) * (self.aperture_max - self.aperture_min)
        xs = []
        ys = []
        zs = []
        for ri in radius.flat:
            x_row = []
            y_row = []
            z_row = []
            for ai in angles.flat:
                xi, yi, zi = self.surf_sag(ri * math.cos(ai), ri * math.sin(ai))
                x_row.append(xi)
                y_row.append(yi)
                z_row.append(zi)
            xs.append(x_row)
            ys.append(y_row)
            zs.append(z_row)
        return np.array(xs), np.array(ys), np.array(zs)
        # surf_cords = [[self.surf_sag(ri * math.cos(ai), ri * math.sin(ai))for ai in angles.flat]for ri in radius.flat]

    def draw(self, axis=None) -> None:
        axis = axis if axis else plt.axes(projection='3d')
        surf = self.surf_shape()
        axis.plot_surface(*surf, linewidths=0.0, antialiased=True, color='blue', edgecolor="none", alpha=0.6)
        axis.set_aspect('equal', 'box')
        axis.set_xlabel("z, [mm]")
        axis.set_ylabel("x, [mm]")
        axis.set_zlabel("y, [mm]")

    def reflect(self, direction: Vector3, origin: Vector3) -> Tuple[float, Vector3, Vector3]:
        t, re, rn = self.intersect_ray(direction, origin)
        return (0.0, origin, re) if t < 0 else (t, re, Vector3.reflect(direction, rn))

    def refract(self, direction: Vector3, origin: Vector3, ri1: float, ri2: float) -> Tuple[float, Vector3, Vector3]:
        t, re, rn = self.intersect_ray(direction, origin)
        return (0.0, origin, re) if t < 0 else (t, re, Vector3.refract(direction, rn, ri1, ri2))

    def trace_ray(self, direction: Vector3, origin: Vector3,
                  wl: float = 450.0, pol: Matrix3 = None) -> Tuple[float, Vector3, Vector3]:
        if not self.material:
            return -1.0, origin, direction
        m_type = self.material['type']
        # match
        # case:

        if m_type == IMAGE_OBJECT:
            return self.intersect_ray(origin, direction)
        if m_type == MIRROR:
            return self.reflect(origin, direction)
        if m_type == GLASS:
            return self.refract(origin, direction, *self.material[GLASS_PARAMS])
        raise ValueError(f"Unknown material of type \"{m_type}\"...")

    @classmethod
    def make_ellipsoid(cls, ellipsoid_semi_axis: Vector3, position: Vector3 = None, angles: Vector3 = None):
        surf = cls()
        surf._direction = 1.0 if ellipsoid_semi_axis.z > 0 else -1.0
        surf._s_type = ELLIPSOIDAL_SURFACE
        surf._m_param.m00 = 1.0 / (ellipsoid_semi_axis.x * ellipsoid_semi_axis.x)
        surf._m_param.m11 = 1.0 / (ellipsoid_semi_axis.y * ellipsoid_semi_axis.y)
        surf._m_param.m22 = 1.0 / (ellipsoid_semi_axis.z * ellipsoid_semi_axis.z)
        surf._b_param = Vector3(0, 0, 0)
        surf._d_param = Vector3(0, 0, -ellipsoid_semi_axis.z)
        surf._a_param.y = 0.9999 * min(surf._a_param.y, abs(ellipsoid_semi_axis.x), abs(ellipsoid_semi_axis.y))
        surf._k_param = -1.0
        if position:
            surf.transform.origin = position
        if angles:
            surf.transform.angles = angles
        return surf

    @classmethod
    def make_sphere(cls, radius: float = 10.0, position: Vector3 = None, angles: Vector3 = None):
        surf = cls.make_ellipsoid(Vector3(radius, radius, radius), position, angles)
        surf._s_type = SPHERICAL_SURFACE
        return surf

    @classmethod
    def make_hyperboloid(cls, hyperboloid_params: Vector3, position: Vector3 = None, angles: Vector3 = None):
        """
        двуполостный гиперболоид
        @param hyperboloid_params:
        @param position:
        @param angles:
        @return:
        """
        surf = cls.make_ellipsoid(hyperboloid_params, position, angles)
        surf._s_type = HYPERBOLIC_SURFACE
        surf._m_param.m00 *= -1
        surf._m_param.m11 *= -1
        surf._direction *= -1.0
        surf._d_param.z *= -1.0
        return surf

    @classmethod
    def make_paraboloid(cls, paraboloid_args: Vector3, position: Vector3 = None, angles: Vector3 = None):
        surf = cls.make_ellipsoid(paraboloid_args, position, angles)
        surf._s_type = PARABOLOID_SURFACE
        surf._m_param.m22 = 0.0
        surf._b_param.z = -paraboloid_args.z
        surf._d_param.z = 0
        surf._k_param = 0
        return surf

    @classmethod
    def make_elliptic_paraboloid(cls, paraboloid_args: Vector3, position: Vector3 = None, angles: Vector3 = None):
        surf = cls.make_ellipsoid(paraboloid_args, position, angles)
        surf._s_type = PARABOLOID_SURFACE
        surf._m_param.m22 = 0.0
        surf._b_param.z = -paraboloid_args.z
        surf._d_param.z *= 0.0
        surf._k_param = 0
        return surf

    @classmethod
    def make_hyperbolic_paraboloid(cls, paraboloid_args: Vector3, position: Vector3 = None, angles: Vector3 = None):
        surf = cls.make_ellipsoid(paraboloid_args, position, angles)
        surf._s_type = PARABOLOID_SURFACE
        surf._m_param.m22 = 0.0
        surf._m_param.m11 *= -1.0
        surf._b_param.z = -paraboloid_args.z
        surf._d_param.z *= 0.0
        surf._k_param = 0
        return surf

    @classmethod
    def make_conic(cls, conic_args: Vector3, position: Vector3 = None, angles: Vector3 = None):
        surf = cls.make_ellipsoid(conic_args, position, angles)
        surf._s_type = CONIC_SURFACE
        surf._m_param.m22 = -conic_args.z * conic_args.z
        surf._direction *= -1.0
        surf._d_param.z = 0.0
        surf._k_param = 0.0
        return surf


def intersect_surface(surface: OpticalSurface, rays_n: int = 5):
    axes = plt.axes(projection='3d')
    xs = np.linspace(0.0, 0.5 * surface.aperture_max, rays_n)
    for xi in xs.flat:
        for yi in xs.flat:
            p0 = Vector3(xi, yi, -5)
            try:
                t, pe, n = surface.intersect_ray(Vector3(0.0, 0.0, 1.0), p0)
                axes.plot([pe.x, pe.x + n.x],
                          [pe.y, pe.y + n.y],
                          [pe.z, pe.z + n.z], 'r')
                axes.plot([p0.x, pe.x],
                          [p0.y, pe.y],
                          [p0.z, pe.z], 'k')
                axes.plot([p0.x, 0.99 * pe.x],
                          [p0.y, 0.99 * pe.y],
                          [p0.z, 0.99 * pe.z], '.r')
            except Exception as error:
                print(error)
                axes.plot([p0.x, p0.x],
                          [p0.y, p0.y],
                          [p0.z, p0.z + 1], 'k')
    surface.draw(axes)
    plt.show()


def surfaces_trace_ray_3d(rd: Vector3, ro: Vector3, surfaces: Iterable[OpticalSurface]) -> \
        Tuple[List[Vector3], List[Vector3]]:
    # дополнительные параметры поверхностей
    """
    Делает трассировку луча через набор сферических поверхностей
    @param ro: начало луча
    @param rd: направление луча
    @param surfaces: список поверхностей
    {'material': 'mirror'} - для зеркала
    или {'material': 'glass', 'glass-params': (1.333, 1.0)} для преломляющей поверхности.
    @return: список точек пересечения с поверхностями и список направления лучей в точках пересечения
    """
    points = [ro]
    directions = [rd]
    for surface_index, surface in enumerate(surfaces):
        material = surface.material
        if not material:
            continue
        try:
            if material[MATERIAL] == MIRROR:
                t, _re, _rd = surface.reflect(directions[-1], points[-1])
                if t < 0:
                    break
                points.append(_re)
                directions.append(_rd)
            if material[MATERIAL] == IMAGE_OBJECT:
                t, _re, _rd = surface.intersect_ray(directions[-1], points[-1])
                if t < 0:
                    break
                points.append(_re)
                directions.append(_rd)
            if material[MATERIAL] == GLASS:
                if GLASS_PARAMS not in material:
                    continue
                ri1, ri2 = material[GLASS_PARAMS]
                t, _re, _rd = surface.refract(directions[-1], points[-1], ri1, ri2)
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


def _cylinder_shape_3d(surface1: OpticalSurface, surface2: OpticalSurface, steps: int = 21):
    steps = max(5, steps)
    da = np.pi * 2 / (steps - 1)
    tc = Transform3d((surface1.transform.origin + surface2.transform.origin) * 0.5,
                     (surface1.transform.scale + surface2.transform.scale) * 0.5,
                     (surface1.transform.angles + surface2.transform.angles) * 0.5)
    aperture = min(surface1.aperture_max, surface2.aperture_max)
    points = [Vector2(aperture * math.cos(da * idx), aperture * math.sin(da * idx)) for idx in range(steps)]
    # ys = [Vector3(aperture * math.cos(da * idx), aperture * math.sin(da * idx), 0.0) for idx in range(steps)]
    x_cords, y_cords, z_cords = [[], []], [[], []], [[], []]
    e0 = tc.front
    for x, y in points:
        p0 = tc.transform_vect(Vector3(x, y, 0.0), 1.0)
        _, t, _ = surface1.intersect_ray(e0, p0)
        _, l, _ = surface2.intersect_ray(-e0, p0)
        # t = tc.inv_transform_vect(t, 1.0)
        # l = tc.inv_transform_vect(l, 1.0)
        x_cords[0].append(t.x)
        x_cords[1].append(l.x)
        y_cords[0].append(t.y)
        y_cords[1].append(l.y)
        z_cords[0].append(t.z)
        z_cords[1].append(l.z)
    aperture = min(surface1.aperture_min, surface2.aperture_min)
    surf_data = {'cyl-surf-1': (np.array(x_cords), np.array(y_cords), np.array(z_cords))}
    if aperture < NUMERICAL_ACCURACY:
        return surf_data
    aperture = min(surface1.aperture_max, surface2.aperture_max)
    xs = [Vector3(aperture * math.cos(da * idx), aperture * math.sin(da * idx), 0.0) for idx in range(steps)]
    ys = [Vector3(aperture * math.cos(da * idx), aperture * math.sin(da * idx), 0.0) for idx in range(steps)]
    x_cords, y_cords, z_cords = [[], []], [[], []], [[], []]
    e0 = tc.transform_vect(Vector3(0.0, 0.0, 1.0), 0.0)
    for x, y in zip(xs, ys):
        p0 = tc.transform_vect(Vector3(x, y, 0.0), 1.0)
        _, t, _ = surface1.intersect_ray(e0, p0)
        _, l, _ = surface1.intersect_ray(-e0, p0)
        x_cords[0].append(t.x)
        x_cords[1].append(l.x)
        y_cords[0].append(t.y)
        y_cords[1].append(l.y)
        z_cords[0].append(t.z)
        z_cords[1].append(l.z)
    return {**surf_data, **{'cyl-surf-2': (np.array(x_cords), np.array(y_cords), np.array(z_cords))}}


def lens_shape_3d(surf1: OpticalSurface, surf2: OpticalSurface, steps: int = 16) -> \
        Dict[str, Tuple[np.ndarray, ...]]:
    xs0, ys0, zs0 = surf1.surf_shape()  # build_shape_3d(r1, s_dia, transform_1, steps)
    xs1, ys1, yz1 = surf2.surf_shape()  # build_shape_3d(r2, s_dia, transform_2, steps)
    # side_surfaces = _cylinder_shape_3d(surf1, surf2, 4 * steps)
    shape = {'front-surf': (xs0, ys0, zs0), 'back-surf': (xs1, ys1, yz1)}
    # shape.update(side_surfaces)
    return shape


def draw_surfaces_scheme(surfaces: Iterable[OpticalSurface], axis=None, steps_a: int = 63, steps_r: int = 63):
    axis = axis if axis else plt.axes(projection='3d')
    iter_surfaces = iter(surfaces)
    surf_index = -1
    while True:
        surf_index += 1
        try:
            surf = next(iter_surfaces)
            material = surf.material
            if not material:
                x, y, z = surf.surf_shape()
                axis.contour3D(x, y, z, antialiased=False, color='grey')
                continue
            if material[MATERIAL] == IMAGE_OBJECT:
                x, y, z = surf.surf_shape(steps_r, steps_a)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='green')
                continue
            if material[MATERIAL] == SOURCE_OBJECT:
                x, y, z = surf.surf_shape(steps_r, steps_a)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='red')
                continue
            if material[MATERIAL] == DUMMY_OBJECT:
                x, y, z = surf.surf_shape(steps_r, steps_a)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='grey')
                continue
            if material[MATERIAL] == MIRROR:
                x, y, z = surf.surf_shape(steps_r, steps_a)
                axis.plot_surface(x, y, z, linewidths=0.0, antialiased=False, color='white')
                continue
            if material[MATERIAL] != GLASS:
                continue
            surf1 = next(iter_surfaces)
            shapes = lens_shape_3d(surf, surf1)
            for s in shapes.values():
                axis.plot_surface(*s, linewidths=0.0, antialiased=True, color='blue', edgecolor="none")
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
    axis.set_zlabel("y, [mm]")
    return axis


def surfaces_intersection_test():
    sign = -1.0
    surf = OpticalSurface.make_sphere(sign * 5, position=Vector3(0, 0, 0), angles=Vector3(0, 0, 0))
    intersect_surface(surf)
    # surf.draw()
    # plt.show()
    # print(surf)
    # print()
    surf = OpticalSurface.make_ellipsoid(Vector3(2, 2, sign * 3), position=Vector3(0, 0, 8), angles=Vector3(1, 2, 3))
    intersect_surface(surf)
    # surf.draw()
    # plt.show()
    # print(surf)
    # print()
    # TODO surface orientation issues
    surf = OpticalSurface.make_conic(Vector3(2, 2, sign * 0.3), position=Vector3(0, 0, 8), angles=Vector3(1, 2, 3))
    intersect_surface(surf)
    # surf.draw()
    # plt.show()
    # print(surf)
    # print()

    surf = OpticalSurface.make_paraboloid(Vector3(2, 2, sign * 0.3), position=Vector3(0, 0, 8), angles=Vector3(1, 2, 3))
    intersect_surface(surf)
    # surf.draw()
    # plt.show()
    # print(surf)
    # print()

    surf = OpticalSurface.make_elliptic_paraboloid(Vector3(2, 2, sign * 0.5), position=Vector3(0, 0, 8),
                                                   angles=Vector3(1, 2, 3))
    intersect_surface(surf)
    # surf.draw()
    # plt.show()
    print(surf)
    print()


def surfaces_tracing_test():
    surfaces_rad = (1e12, -350, -350, 1e12, -350, 350, 550, 350, 1e12)  # : Iterable[float]
    aperture_rad = (50, 50, 50, 55, 50, 50, 50, 20, 20)  # : Iterable[float]
    surfaces_ang = (Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0),
                    Vector3(0.0, 90.0, 0.0))
    surfaces_pos = (Vector3(70 + -50, 0.0, 0.0),
                    Vector3(70 + -15, 0.0, 0.0),
                    Vector3(70 + -5, 0.0, 0.0),
                    Vector3(70 + 0, 0.0, 0.0),
                    Vector3(70 + 0, 0.0, 0.0),
                    Vector3(70 + 30, 0.0, 0.0),
                    Vector3(70 + 125, 0.0, 0.0),
                    Vector3(70 + 30.1, 0.0, 0.0),
                    Vector3(70 + 150, 0.0, 0.0))
    surfaces_mat = [{MATERIAL: SOURCE_OBJECT},
                    {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.66)},
                    {MATERIAL: GLASS, GLASS_PARAMS: (1.66, 1.0)},
                    {MATERIAL: DUMMY_OBJECT},
                    {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.333)},
                    {MATERIAL: GLASS, GLASS_PARAMS: (1.333, 1.0)},
                    {MATERIAL: MIRROR},
                    {MATERIAL: MIRROR},
                    {MATERIAL: IMAGE_OBJECT}]
    surfaces = []
    for sr, ar, sa, sp, sm in zip(surfaces_rad, aperture_rad, surfaces_ang, surfaces_pos, surfaces_mat):
        surf = OpticalSurface.make_sphere(-sr)
        surf.transform.origin = sp
        surf.transform.angles = sa
        surf.material.update(sm)
        surf.aperture = ar
        surfaces.append(surf)
    axis = plt.axes(projection='3d')
    draw_surfaces_scheme(surfaces, axis=axis)  # [0:5])

    for rd, ro in rect_source():
            positions, directions = surfaces_trace_ray_3d(rd, ro, surfaces)
            xs = [v.x for v in positions]
            ys = [v.y for v in positions]
            zs = [v.z for v in positions]
            axis.plot(xs, ys, zs, 'r')
    plt.show()
