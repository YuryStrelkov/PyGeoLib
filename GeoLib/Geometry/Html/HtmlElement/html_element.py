import os
from typing import Set, Iterable, Generator, Union, Any, List
from ..AttributesContainer import AttributesContainer
from ..CSS import CssElementStyle


class HtmlClassList:
    __slots__ = '_attributes'

    def __init__(self):
        self._attributes: Set[str] = set()

    def __str__(self) -> str:
        return ' '.join(f"{v}"for v in self.attributes)

    def __iter__(self) -> Generator[str, None, None]:
        return self.attributes

    @property
    def attributes(self) -> Generator[str, None, None]:
        yield from self._attributes

    def add_attribute(self, _class: str) -> None:
        self._attributes.add(_class)

    def del_attribute(self, _class: str) -> None:
        self._attributes.remove(_class)


class HtmlElement(AttributesContainer):
    _ELEMENTS_IDS = set()
    _ELEMENT_ID: int = 0

    @staticmethod
    def _get_id() -> int:
        if len(HtmlElement._ELEMENTS_IDS) != 0:
            return HtmlElement._ELEMENTS_IDS.pop()
        _ID = HtmlElement._ELEMENT_ID
        HtmlElement._ELEMENT_ID += 1
        return _ID

    @staticmethod
    def _release_id(_id: int) -> None:
        HtmlElement._ELEMENTS_IDS.add(_id)

    def __init__(self, open_tag: str = None, close_tag: str = None, *,
                 parent: Union['HtmlElement', None] = None,
                 inner_html: str = "", class_list: Iterable[Any] = None):
        super().__init__()
        self._level: int = 0
        self._parent: Union['HtmlElement', None] = None
        self._children: List = list()
        self._open_tag: str = open_tag
        self._close_tag: str = close_tag
        self._inner_html: str = inner_html
        self._element_id: int = HtmlElement._get_id()
        if class_list:
            for _class in class_list:
                self.add_to_class_list(str(_class))
        if parent:
            self._set_parent(parent)

    def __del__(self):
        HtmlElement._release_id(self.element_id)

    def _render_attributes(self) -> str:
        return ' '.join(f"{key}=\"{str(val)}\""for key, val in self.attributes if val)if self.has_attributes else ""

    def _render_element(self) -> str:
        if attributes := self._render_attributes():
            return f"id=\"element-{self._element_id}\" {attributes}"
        else:
            return f"id=\"element-{self._element_id}\""

    def _render_inline(self) -> str:
        indent = '\t' * self.level
        opn, cls = self.open_tag, self.close_tag
        nl = '\n'
        if opn and cls:
            return f"{indent}<{opn} {self._render_element()}>{self.inner_html}</{cls}>"
        if opn:
            return f"{indent}<{opn} {self._render_element()}>" \
                   f"{f'{nl}{indent}<text>{self.inner_html}</text>' if self.inner_html else ''}"
        return f"{indent}<div {self._render_element()}>{self.inner_html}</div>"

    def render_element(self) -> str:
        if not self.has_children:
            return self._render_inline()
        indent = '\t' * self.level
        children = '\n'.join(v.render_element() for v in self._children)
        inner_html = f" {self.inner_html}" if len(self.inner_html) else ""
        return f"{indent}<{self.open_tag} {self._render_element()}>{inner_html}\n{children}\n{indent}</{self.close_tag}>"
        # o, c = self.open_tag, self.close_tag
        # if o and c:
        # if o:
        #     return f"{indent}<{o} {self._render_element()}>{inner_html}{indent}\n{children}"
        # return f"{indent}<div {self._render_element()}>\n{inner_html}{children}\n{indent}</div>"

    def __str__(self) -> str:
        return self.render_element()

    def _set_parent(self, parent: Union['HtmlElement', None]) -> bool:
        if self.has_parent:
            self._parent._children.remove(self)
        self._parent = parent
        if self.has_parent:
            self._parent._children.append(self)
            self._level = self.parent.level + 1
            nodes = list(self.children)
            while len(nodes):
                node: 'HtmlElement' = nodes.pop(0)
                node._level = node.parent.level + 1
                nodes.extend(node.children)
        return self.has_parent

    @property
    def level(self) -> int:
        return self._level

    @property
    def parent(self) -> Union['HtmlElement', None]:
        return self._parent

    @parent.setter
    def parent(self, value: Union['HtmlElement', None]) -> None:
        self._set_parent(value)

    @property
    def has_parent(self) -> bool:
        return bool(self._parent)

    @property
    def has_children(self) -> bool:
        return len(self._children) != 0

    @property
    def children(self) -> Iterable['HtmlElement']:
        yield from self._children

    def add_child(self, node: 'HtmlElement') -> bool:
        if not isinstance(node, HtmlElement):
            return False
        if node in self._children:
            return False
        return node._set_parent(self)

    def remove_child(self, node: 'HtmlElement') -> bool:
        if not isinstance(node, HtmlElement):
            return False
        if node.parent != self:
            return False
        self._children.remove(node)
        node._parent = None
        return True

    @property
    def open_tag(self) -> str:
        return self._open_tag

    @property
    def close_tag(self) -> str:
        return self._close_tag

    @property
    def element_id(self) -> int:
        return self._element_id

    # @element_id.setter
    # def element_id(self, value: str) -> None:
    #     if value and isinstance(value, str):
    #         if self.has_parent:
    #             if any(c.element_id == value for c in self.parent.children):
    #                 print("Parent node already contains node with the same id")
    #                 return
    #         self._element_id = value

    @property
    def inner_html(self) -> str:
        return self._inner_html

    @inner_html.setter
    def inner_html(self, value: str) -> None:
        if value and isinstance(value, str):
            self._inner_html = value

    @property
    def style(self) -> CssElementStyle:
        return self._get_attribute("style", CssElementStyle)

    @property
    def class_list(self) -> HtmlClassList:
        return self._get_attribute("class", HtmlClassList)

    def add_to_class_list(self, value: str) -> None:
        self.class_list.add_attribute(value)

    def remove_from_class_list(self, value: str) -> None:
        self.class_list.del_attribute(value)
