import dataclasses
import os
from typing import Dict, Tuple, List, Union
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from Geometry import Vector3, Vector2, BoundingBox, Transform3d, read_obj_files


# def draw(self, axis=None, show: bool = True):
#     axis = axis if axis else plt.axes(projection='3d')
#     faces = self.faces_array
#     vertices = self.vertices_array
#     axis.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, shade=True)
#     axis.set_xlabel("x, [mm]")
#     axis.set_ylabel("y, [mm]")
#     axis.set_zlabel("z, [mm]")
#     if show:
#         axis.set_aspect('equal', 'box')
#         plt.show()
#     return axis


    # @property
    # def sideways_vertices(self) -> Tuple[Tuple[Vector3, ...], ...]:
    #     """
    #                 UP(max.y)
    #                    ||
    #     LEFT(min.x) ======== RIGHT(max.x)
    #                    ||
    #                 DOWN(min.y)
    #     :return:
    #     """
    #     return self.left_vertices, self.right_vertices, self.top_vertices, self.down_vertices
    #
    # @property
    # def left_vertices(self) -> Tuple[Vector3, ...]:
    #     _min = self.bounds.min
    #     return tuple(v for v in self.vertices if abs(v.x - _min.x) < 1e-6)
    #
    # @property
    # def right_vertices(self) -> Tuple[Vector3, ...]:
    #     _max = self.bounds.max
    #     return tuple(v for v in self.vertices if abs(v.x - _max.x) < 1e-6)
    #
    # @property
    # def top_vertices(self) -> Tuple[Vector3, ...]:
    #     _max = self.bounds.max
    #     return tuple(v for v in self.vertices if abs(v.z - _max.z) < 1e-6)
    #
    # @property
    # def down_vertices(self) -> Tuple[Vector3, ...]:
    #     _min = self.bounds.min
    #     return tuple(v for v in self.vertices if abs(v.z - _min.z) < 1e-6)


def _is_shapes_close(shape1, shape2) -> bool:
    integral1 = sum(pt for pt in shape1).magnitude
    integral2 = sum(pt for pt in shape2).magnitude
    print(f" abs(integral1 - integral2 ) : { abs(integral1 - integral2 )}")
    return abs(integral1 - integral2 ) < 1.0
    # return all(all(abs(x - y) < 10.0 for x, y in zip(p1, p2)) for p1, p2 in zip(shape1, shape2))

#
# def check_connectable(target: Union[ObjMesh, np.ndarray], pretender: Union[ObjMesh, np.ndarray]):
#     """
#     :param target:
#     :param pretender:
#     :return:
#     Tuple of pairs:
#         Angle id: (ex. 0 - zero angle, 1 - 90 degrees angle, ...)
#     Tuple of connection rules:
#         Target left side - pretender right side. Connectable if not equals to -1
#         Target right side - pretender left side. Connectable if not equals to -1
#         Target top side - pretender bottom side. Connectable if not equals to -1
#         Target bottom side - pretender top side. Connectable if not equals to -1
#     r - right
#     l - left
#     t - top
#     d - down
#                     angle = 0.0              |           angle = 90.0          |           angle = 180.0         |            angle = 270.0
#     rule = ( l <- r, r <- l, d <- u, u <- d  | l <- r, r <- l, d <- u, u <- d  | l <- r, r <- l, d <- u, u <- d  |  l <- r, r <- l, d <- u, u <- d )
#     """
#     if isinstance(target, ObjMesh):
#         if target.bounds.size != target.bounds.size:
#             return ()
#         t_l = target.left_vertices
#         t_r = target.right_vertices
#         t_u = target.top_vertices
#         t_d = target.down_vertices
#
#         p_l = pretender.left_vertices
#         p_r = pretender.right_vertices
#         p_u = pretender.top_vertices
#         p_d = pretender.down_vertices
#     elif isinstance(target, np.ndarray):
#         if pretender.shape != target.shape:
#             return ()
#         rows, cols, _ = pretender.shape
#         rows, cols =   rows * 0.5 - 0.5, cols * 0.5 - 0.5
#         t_l = tuple(Vector3(-cols, idx - rows, sum(v for v in color)) for idx, color in enumerate(target[:, 0, :] ))
#         t_r = tuple(Vector3( cols, idx - rows, sum(v for v in color)) for idx, color in enumerate(target[:, -1, :]))
#         t_u = tuple(Vector3( idx - cols, rows, sum(v for v in color)) for idx, color in enumerate(target[0, :, :] ))
#         t_d = tuple(Vector3( idx - cols,-rows, sum(v for v in color)) for idx, color in enumerate(target[-1, :, :]))
#
#         p_l = tuple(Vector3(-cols, idx - rows, sum(v for v in color)) for idx, color in enumerate(pretender[:, 0, :] ))
#         p_r = tuple(Vector3( cols, idx - rows, sum(v for v in color)) for idx, color in enumerate(pretender[:, -1, :]))
#         p_u = tuple(Vector3( idx - cols, rows, sum(v for v in color)) for idx, color in enumerate(pretender[0, :, :] ))
#         p_d = tuple(Vector3( idx - cols,-rows, sum(v for v in color)) for idx, color in enumerate(pretender[-1, :, :]))
#     else:
#         return ()
#
#     angle = 0
#     transform = Transform3d()
#     transform.angles = Vector3(0.0, 90.0, 0.0)
#     """
#        l, u, r, d
#     """
#     connectivity = [1 if _is_shapes_close(t_l, p_r) else 0,
#                     1 if _is_shapes_close(t_r, p_l) else 0,
#                     1 if _is_shapes_close(t_d, p_u) else 0,
#                     1 if _is_shapes_close(t_u, p_d) else 0]
#
#     p_l = tuple(transform.transform_vect(v) for v in p_l)
#     p_r = tuple(transform.transform_vect(v) for v in p_r)
#     p_u = tuple(transform.transform_vect(v) for v in p_u)
#     p_d = tuple(transform.transform_vect(v) for v in p_d)
#     """
#                 LEFT(max.y)
#                    ||
#     DOWN(min.x) ======== UP(max.x)
#                    ||
#                 RIGHT(min.y)
#     l, u, r, d ->
#     d, l, u, r
#     """
#     angle += 1
#     connectivity.extend((1 if _is_shapes_close(t_l, p_u) else 0,
#                          1 if _is_shapes_close(t_r, p_d) else 0,
#                          1 if _is_shapes_close(t_d, p_l) else 0,
#                          1 if _is_shapes_close(t_u, p_r) else 0))
#
#     p_l = tuple(transform.transform_vect(v) for v in p_l)
#     p_r = tuple(transform.transform_vect(v) for v in p_r)
#     p_u = tuple(transform.transform_vect(v) for v in p_u)
#     p_d = tuple(transform.transform_vect(v) for v in p_d)
#     angle += 1
#     # d, l, u, r ->
#     # r, d, l, u
#     connectivity.extend((1 if _is_shapes_close(t_l, p_l) else 0,
#                          1 if _is_shapes_close(t_r, p_r) else 0,
#                          1 if _is_shapes_close(t_d, p_d) else 0,
#                          1 if _is_shapes_close(t_u, p_u) else 0))
#
#     p_l = tuple(transform.transform_vect(v) for v in p_l)
#     p_r = tuple(transform.transform_vect(v) for v in p_r)
#     p_u = tuple(transform.transform_vect(v) for v in p_u)
#     p_d = tuple(transform.transform_vect(v) for v in p_d)
#     angle += 1
#     # r, d, l, u ->
#     # u, r, d, l
#     connectivity.extend((1 if _is_shapes_close(t_l, p_d) else 0,
#                          1 if _is_shapes_close(t_r, p_u) else 0,
#                          1 if _is_shapes_close(t_d, p_r) else 0,
#                          1 if _is_shapes_close(t_u, p_l) else 0))
#     return tuple(connectivity)
#
#
# def build_connections_table(objects: Union[Dict[str, ObjMesh], Dict[str, np.ndarray]]):
#     connection_rules = {}
#     for o_key, o_mesh in objects.items():
#         rules = {}
#         connection_rules.update({o_key: rules})
#         for o_key_c, o_mesh_c in objects.items():
#             rules.update({o_key_c: check_connectable(o_mesh, o_mesh_c)})
#     return connection_rules

#
# def load_image_tiles(dir_path: str) ->  Dict[str, np.ndarray]:
#     return dict((img.name, np.array(Image.open(img.path))) for img in os.scandir(dir_path))
#
#
# @dataclasses.dataclass()
# class TilesMap:
#     tiles: Dict[str, np.ndarray]
#     tiles_ids: Dict[str, int]
#     connection_rules: Dict[str, Dict[str, Tuple[int, ...]]]
#
#     def __init__(self, src_dir: str):
#         self.tiles = load_image_tiles(src_dir)
#         self.tiles_ids = {src: i for i, src in enumerate(self.tiles)}
#         self.connection_rules = build_connections_table(self.tiles)  # check_connectable(objects['corner'], objects['wall'])
#
#     def __str__(self):
#         tiles_str = ',\n'.join(f"\t{{\n\t\t\"tile-id\":{t_id},\n\t\t\"tile-src\":\"{t_name}\"\n\t}}"
#                                for t_id, t_name in enumerate(tmap.tiles))
#         return f"{{\n" \
#                f"\t\"tiles\":[\n{tiles_str}\n\t],\n" \
#                f"\t\"rules\":[\n{self._tile_connection_rules()}\n\t]\n" \
#                f"}}"
#
#     @staticmethod
#     def _tile_rule(tile_id: int, connections: Tuple[int, ...]) -> str:
#         return f"\t\t{{\n" \
#                f"\t\t\t\"tile-id\": {tile_id},\n" \
#                f"\t\t\t\"rules\"  : [{', '.join(str(idx) for idx in connections)}]\n" \
#                f"\t\t}}"
#
#     def _tile_rules(self, tile_id: str) -> str:
#         connections = self.connection_rules[tile_id]
#         nl = ',\n'
#         return f"\t{{\n" \
#                f"\t\t\"tile-id\": {self.tiles_ids[tile_id]},\n" \
#                f"\t\t\"rules\"  : [\n{nl.join(TilesMap._tile_rule(self.tiles_ids[idx], rules) for idx, rules in connections.items())}\n\t\t]\n" \
#                f"\t}}"
#
#     def _tile_connection_rules(self) -> str:
#         nl = ',\n'
#         return f"{nl.join(self._tile_rules(t_id) for t_id in self.connection_rules)}"


if __name__ == "__main__":
    objects = read_obj_files("tiles-set.obj")
    axis = None
    for o in objects:
        objects[o].align_2_center()
        print(f"mesh: {o}\n{objects[o].bounds}")

    axis = axis if axis else plt.axes(projection='3d')
    border = objects['wall'].border
    # sides = objects['wall'].sideways_vertices
    # l_side1 = objects['corner'].left_vertices
    # for p1, p2 in zip(l_side1[:-1], l_side1[1:]):
    #     plt.plot((p1.x, p2.x),
    #              (p1.y, p2.y),
    #              (p1.z, p2.z), 'og')
    #
    # l_side2 = objects['wall'].right_vertices
    # for p1, p2 in zip(l_side2[:-1], l_side2[1:]):
    #     plt.plot((p1.x, p2.x),
    #              (p1.y, p2.y),
    #              (p1.z, p2.z), '.r')

    bounds =  objects['wall'].bounds
    b_min = bounds.min
    b_max = bounds.max
    delta = Vector3(bounds.size.x, 0, 0)
    # print(f"integral value: {sum((p1 - p2 + delta) for p1, p2 in zip(l_side1, l_side2)).magnitude }")
    # tmap = TilesMap("src-tex/min-tiles")
    #
    # # tiles = load_image_tiles("src-tex/tiles")
    #
    # # rules = build_connections_table(tiles)  # check_connectable(objects['corner'], objects['wall'])
    # tiles_ids = {src: i for i, src in enumerate(tmap.tiles)}
    # with open('tile-map.json', 'wt') as t_map:
    #     print(tmap, file=t_map)
    # #     print("{", file=t_map)
    #
    # for index, (key, item) in enumerate(objects.items()):
    #     item.draw(axis, len(objects) - 1 == index)

    objects['corner'].draw(axis, False)
    objects['wall'].draw(axis, True)
