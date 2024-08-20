from ...common import NUMERICAL_ACCURACY
from ...Vectors.vector2 import Vector2
from typing import Tuple, List, Dict, Union
from matplotlib import pyplot as plt
from functools import reduce


def polygon_bounds(polygon: List[Vector2]) -> Tuple[Vector2, Vector2]:
    min_b = Vector2(1e12, 1e12)
    max_b = Vector2(-1e12, -1e12)
    for p in polygon:
        if p.x > max_b.x:
            max_b.x = p.x
        if p.y > max_b.y:
            max_b.y = p.y
        if p.x < min_b.x:
            min_b.x = p.x
        if p.y < min_b.y:
            min_b.y = p.y
    return min_b, max_b


def polygon_area(polygon: List[Vector2]) -> float:
    area: float = 0.0
    for i in range(len(polygon) - 1):
        area += Vector2.cross(polygon[i], polygon[i + 1])
    return area * 0.5


def clamp_index(index: int, index_min: int, index_max: int) -> int:
    return max(min(index, index_max), index_min)


class Polygon:
    __slots__ = ('vertices', 'uvs', 'faces')

    def __init__(self):
        self.vertices: Tuple[Vector2]
        self.faces: Tuple[Tuple[int, ...]]
        self.uvs: Tuple[Vector2]

    def draw(self, axis=None, color="k"):
        axis = axis if axis else plt.axes(projection='3d')
        for face in self.faces:
            i1, i2, i3 = face
            p1, p2, p3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
            axis.plot((p1.x, p2.x, p3.x, p1.x), (p1.y, p2.y, p3.y, p1.y), color)
        return axis


def _pt_within_tris(pt: Vector2, p1: Vector2, p2: Vector2, p3: Vector2) -> bool:
    s0 = 0.5 * abs(Vector2.cross(p1 - p2, p1 - p3))
    s1 = 0.5 * abs(Vector2.cross(pt - p1, pt - p3))
    s2 = 0.5 * abs(Vector2.cross(pt - p2, pt - p3))
    s3 = 0.5 * abs(Vector2.cross(pt - p1, pt - p2))
    return abs(s0 - s1 - s2 - s3) < NUMERICAL_ACCURACY


def triangulate_polygon(polygon: Union[List[Vector2], Tuple[Vector2, ...]]) -> Polygon:
    """
    This function will triangulate any polygon and return a list of triangles

        Parameters:
            polygon (tuple): A tuple of tuples with the points of the polygon
        Returns:
            Polygon:
    """

    """
    Explanation:
    ============

        This function will triangulate any polygon and return a list of triangles
        The algorithm is based on the ear clipping method, which is a simple method

    Algorithm:
    ==========

        1. Find the internal angle of each vertex
        2. Check if the angle is less than 180 degrees, skip if not
        3. Check if there are any polygon points inside the triangle formed by the vertex and its neighbors
        4. If there are no points inside the triangle, then the vertex is an ear
        5. If the vertex is an ear, add the triangle to the list of triangles
        6. Remove the ear and repeat the process until there are no more ears

    References:
    ===========
        https://en.wikipedia.org/wiki/Polygon_triangulation#Ear_clipping_method
        https://www.youtube.com/watch?v=QAdfkylpYwc
    """

    clock_wise = sum(Vector2.cross(a, b) for a, b in zip(polygon[:-1], polygon[1:])) > 0.0
    delta = Vector2.distance(polygon[0], polygon[-1])

    if clock_wise:
        vertices = list(reversed(polygon)) if delta > NUMERICAL_ACCURACY else list(reversed(polygon[:-1]))
    else:
        vertices = list(polygon) if delta > NUMERICAL_ACCURACY else list(polygon[:-1])

    triangles_founded = True
    polygon_vertices = []
    polygon_indices  = []
    indices_shift = -1
    point_per_index: Dict[Vector2, int] = {}
    # While there are triangles to be found
    while triangles_founded:
        triangles_founded = False
        for index, _ in enumerate(vertices):
            v_prev = vertices[index - 1]
            v_curr = vertices[index]
            v_next = vertices[(index + 1) % len(vertices)]  # using mod to avoid index out of range
            # Get Vector from prev_vertice to vertice
            vector1 = v_prev - v_curr
            # Get Vector from vertice to next_vertice
            vector2 = v_next - v_curr
            # Get internal angle
            if Vector2.cross(vector1, vector2) <= 0.0:
                # Skip because angle is greater than 180
                continue
            # Build a triangle with the three vertices
            triangle = (v_prev, v_curr, v_next)
            # Get vertices that are not part of the triangle
            # Check if there is a vertice inside the triangle
            sequence = (_pt_within_tris(pt, *triangle) for pt in polygon if pt not in triangle)
            # If are not points inside the triangle
            if not reduce(lambda a, b: a | b, sequence, False):
                tris = []
                for pt in triangle:
                    if pt not in point_per_index:
                        point_per_index.update({pt: (indices_shift := 1 + indices_shift)})
                        polygon_vertices.append(pt)
                    tris.append(point_per_index[pt])
                polygon_indices.append(tris)
                triangles_founded |= True
                del vertices[index]
                break
    polygon_shape: Polygon = Polygon()
    poly_min, poly_max = polygon_bounds(polygon_vertices)
    duv = poly_max - poly_min
    polygon_shape.vertices = tuple(p for p in polygon_vertices)
    polygon_shape.uvs = tuple((p - poly_min) / duv for p in polygon_vertices)
    polygon_shape.faces = tuple(tris for tris in polygon_indices)
    return polygon_shape


def triangulate_polygons(polygons: List[List[Vector2]]) -> List[Polygon]:
    return [triangulate_polygon(polygon) for polygon in polygons]
