import dataclasses
from typing import Tuple, Union, List, Generator
import random

C_ERROR = '\033[31m'
C_MOUNTAIN = '\033[90m'
C_LAND = '\033[32m'
C_SAND = '\033[33m'
C_SEA = '\033[34m'
C_END = '\033[0m'

NODES_MAP_COLORED = {-1: f"{C_ERROR}X{C_END}",
                     0: f"{C_MOUNTAIN}[-]{C_END}",
                     1: f"{C_LAND}[-]{C_END}",
                     2: f"{C_SAND}[-]{C_END}",
                     3: f"{C_SEA}[-]{C_END}"}

NODES_MAP_HTML_COLORED = {
                          0: "rgb(125 130 135)",
                          1: "rgb(0 200 0)",
                          2: "rgb(255 250 0)",
                          3: "rgb(50 100 255)"}

NODES_MAP_NAMES =  {
                          0: "MOUNTAIN",
                          1: "LAND",
                          2: "COAST",
                          3: "SAND"}

NODES_MAP = 'MLCS'
NODES_MAP_MOUNTAIN_ID = 0
NODES_MAP_LAND_ID = 1
NODES_MAP_COAST_ID = 2
NODES_MAP_SEA_ID = 3


ALL_VARIANTS = (NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID, NODES_MAP_COAST_ID, NODES_MAP_SEA_ID)


CONNECTION_RULES = {
    # top, down, right, left
    NODES_MAP_MOUNTAIN_ID:
        ((NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID),  # top
         (NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID),  # down
         (NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID),  # right
         (NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID)), # left
    NODES_MAP_LAND_ID:
        ((NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID, NODES_MAP_COAST_ID),  # top
         (NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID, NODES_MAP_COAST_ID),  # down
         (NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID, NODES_MAP_COAST_ID),  # right
         (NODES_MAP_MOUNTAIN_ID, NODES_MAP_LAND_ID, NODES_MAP_COAST_ID)),  # left
    NODES_MAP_COAST_ID:
        ((NODES_MAP_LAND_ID, NODES_MAP_COAST_ID, NODES_MAP_SEA_ID),  # top
         (NODES_MAP_LAND_ID, NODES_MAP_COAST_ID, NODES_MAP_SEA_ID),  # down
         (NODES_MAP_LAND_ID, NODES_MAP_COAST_ID, NODES_MAP_SEA_ID),  # right
         (NODES_MAP_LAND_ID, NODES_MAP_COAST_ID, NODES_MAP_SEA_ID)),  # left
    NODES_MAP_SEA_ID:
        ((NODES_MAP_COAST_ID, NODES_MAP_SEA_ID),  # top
         (NODES_MAP_COAST_ID, NODES_MAP_SEA_ID),  # down
         (NODES_MAP_COAST_ID, NODES_MAP_SEA_ID),  # right
         (NODES_MAP_COAST_ID, NODES_MAP_SEA_ID)),  # left
    }


@dataclasses.dataclass
class WFC2D_Node:
    def __init__(self, row: int, col: int):
        self._node_id: Tuple[int, int] = (row, col)
        self._states: List[int] = list(ALL_VARIANTS)
        self.node_up: Union[WFC2D_Node, None] = None
        self.node_down: Union[WFC2D_Node, None] = None
        self.node_right: Union[WFC2D_Node, None] = None
        self.node_left: Union[WFC2D_Node, None] = None

    @property
    def nodes(self) -> Generator['WFC2D_Node', None, None]:
        yield self.node_up
        yield self.node_down
        yield self.node_right
        yield self.node_left

    def __str__(self) -> str:
        return f"| states : [{', '.join(str(s) for s in self._states)}],\n" \
               f"| node_id: [{', '.join(str(s) for s in self._node_id)}],\n" \
               f"| \tnode_up   : [{', '.join(str(s) for s in self.node_up.node_id) if self. node_up else 'None'}],\n" \
               f"| \tnode_down : [{', '.join(str(s) for s in self.node_down.node_id) if self.node_down else 'None'}],\n" \
               f"| \tnode_right: [{', '.join(str(s) for s in self.node_right.node_id) if self.node_right else 'None'}],\n" \
               f"| \tnode_left : [{', '.join(str(s) for s in self.node_left.node_id) if self. node_left else 'None'}]\n\n"

    @property
    def node_id(self) ->  Tuple[int, int]:
        return self._node_id

    def _update_connection(self, cell_state: int, direction: int) -> None:
        if cell_state not in CONNECTION_RULES:
            raise ValueError(f"cell_state not in cell_states\nstate:{cell_state}\ncell:\n{self}")
        if self.is_collapsed:
            return
        rules = CONNECTION_RULES[cell_state][direction]
        self._states = list(item for item in self._states if item in rules)

    @property
    def is_collapsed(self) -> bool:
        return self.states_count == 1  # or self.collapsed_value == -1

    @property
    def collapsed_value(self) -> int:
        return self._states[0]  # -1 if self.states_count == 0 else self._states[0]

    @property
    def states_count(self) -> int:
        return len(self._states)

    def collapse(self, cell_state: int = -1) -> int:
        """
        Collapse cell by it value
        :param cell_state:
        :return:
        """
        self._states = [self._states[random.randint(0, len(self._states) - 1)]] if cell_state == -1 else [cell_state]
        for index, node in enumerate(self.nodes):
            if not node:
                continue
            node._update_connection(self.collapsed_value, index)
        return 0


class WFC2D_Map:
    """
    Naive wave function collapse algorythm implementation with fixed tile set
    """
    def __init__(self, rows: int = 32, cols: int = 32):
        self._rows, self._cols = max(3, rows), max(3, cols)
        self._nodes = ()

    def rebuild(self):
        self._nodes: Tuple[WFC2D_Node, ...] = tuple(WFC2D_Node(*divmod(index, self.cols))
                                                    for index in range(self.rows * self.cols))
        for node in self._nodes:
            row, col = node.node_id
            node.node_up = None if row + 1 >= self.rows else self._nodes[(row + 1) * self.cols + col]
            node.node_down = None if row - 1 < 0 else self._nodes[(row - 1) * self.cols + col]
            node.node_right = None if col + 1 >= self.cols else self._nodes[row * self.cols + col + 1]
            node.node_left = None if col - 1 < 0 else self._nodes[row * self.cols + col - 1]

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def __str__(self) -> str:
        return f"{''.join(str(node) for node in self._nodes)}"

    def __call__(self, row: int = 0, col: int = 0, init_state: int = 0):
        self.rebuild()
        node = self._nodes[row * self.cols + col]
        node.collapse(init_state)
        neighbours = [n for n in node.nodes if n]
        while len(neighbours) != 0:
            val = 0xfffffff
            node = None
            for n in neighbours:
                if n.is_collapsed:
                    continue
                if n.states_count > val:
                    continue
                val = n.states_count
                node = n
            node.collapse()
            for n in node.nodes:
                if not n:
                    continue
                neighbours.append(n)
            neighbours = [n for n in neighbours if not n.is_collapsed]

    def render(self) -> str:
        def render_row(row: int) -> str:
            return ''.join(str(NODES_MAP_COLORED[self._nodes[i].collapsed_value]) for i in range(row * self.cols, (row + 1) * self.cols))
        return '\n'.join(render_row(row)for row in range(self.rows))

    def render_html(self, width: int = 1000, height: int = 1000):
        width = width if(c_w := width // self.cols) != 0 else self.cols
        height = height if(c_h := height // self.rows) != 0 else self.rows
        header = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Tiled map</title>
                <style>
                    html, body{{
                        width: {width}px;
                        height: {height}px;
                        margin: 0;
                        padding: 0;
                        background-color: black;
                    }}
                    .tooltip {{}}
                    .tooltip .tooltip-text {{
                      visibility: hidden;
                      width: 120px;
                      background-color: rgba(128, 132, 136, 0.666);
                      color: #fff;
                      text-align: center;
                      border-radius: 6px;
                      padding: 5px 0;
                      position: absolute;
                      z-index: 1;
                    }}
                    .tooltip:hover .tooltip-text {{
                      visibility: visible;
                    }}
                </style>
            </head>
            <body>\n
        """
        footer = """
            </body>
        </html>
        """
        str_nodes = []
        for index, node in enumerate(self._nodes):
            row, col = divmod(index, self.cols)
            row *= c_w
            col *= c_h
            row += 2
            col += 2
            color = NODES_MAP_HTML_COLORED[node.collapsed_value]
            tooltip = NODES_MAP_NAMES[node.collapsed_value]
            str_nodes.append(f"\t\t\t<div  class=\"tooltip\" style=\"position: absolute; "
                             f"top: {row}px; "
                             f"left: {col}px; "
                             f"width: {c_w - 2}px; "
                             f"height: {c_h - 2}px; "
                             f"background-color: {color};\">"
                             f"<span class=\"tooltip-text\">{tooltip}</span>"
                             f"</div>")
        nl = '\n'
        return f"{header}{nl.join(l for l in str_nodes)}{footer}"


if __name__ == "__main__":
    wfc_map = WFC2D_Map()
    wfc_map()
    with open('tiles-map.html', 'wt') as tl:
        print(wfc_map.render_html(), file=tl)
    print(wfc_map.render())

