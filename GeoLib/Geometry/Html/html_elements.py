from typing import Set, Any, Dict, Generator

from .HtmlElement import *


class ChildFreeElement(HtmlElement):
    def add_child(self, node: 'HtmlElement') -> bool:
        print(f"Element \"{self.open_tag}\" unable to own the children...")
        return self.parent.add_child(node) if self.has_parent else False

    def render_element(self) -> str:
        indent = '\t' * self.level
        inner_html = f" property=\"{self.inner_html}\"" if len(self.inner_html) else ""
        return f"{indent}<{self.open_tag}{inner_html} {self._render_element()}>"


class HtmlList:
    ...


class HtmlOrderedList:
    ...


class HtmlArticle:
    ...


class HtmlTable:
    ...


class HtmlLink(ChildFreeElement):
    def __init__(self, parent: HtmlElement):
        super().__init__("link", parent=parent)

    @property
    def href(self) -> str:
        return super()._get_attribute("href", lambda: "")

    @href.setter
    def href(self, value: str) -> None:
        super().add_attribute("href", str(value))

    @property
    def rel(self) -> str:
        return super()._get_attribute("rel", lambda: "")

    @rel.setter
    def rel(self, value: str) -> None:
        super().add_attribute("rel", str(value))


class HtmlScript(ChildFreeElement):
    def __init__(self, parent: HtmlElement):
        super().__init__("script", "script", parent=parent)
        self.code = f"alert(\"hello from script of node with id: {self.element_id}!!!\");"
        self.type = "inline"

    def render_element(self) -> str:
        indent = '\t' * self.level
        inner_html = f" code=\"{self.inner_html}\"" if len(self.inner_html) else "code=\"\""
        return f"{indent}<{self.open_tag}{inner_html} {self._render_element()}>"

    @property
    def code(self) -> str:
        return self.inner_html

    @code.setter
    def code(self, value: str) -> None:
        self.inner_html = value

    @property
    def source(self) -> str:
        return super()._get_attribute("src", lambda: "")

    @source.setter
    def source(self, value: str) -> None:
        super().add_attribute("src", str(value))

    @property
    def type(self) -> str:
        return super()._get_attribute("type", lambda: "inline")

    @type.setter
    def type(self, value: str) -> None:
        super().add_attribute("type", str(value))


class HtmlMeta(ChildFreeElement):
    def __init__(self, parent: HtmlElement):
        super().__init__("meta", parent=parent)


class HtmlStyle(ChildFreeElement):
    def __init__(self, parent: HtmlElement):
        super().__init__("style", "style", parent=parent)


class HtmlTitle(HtmlElement):
    def __init__(self, parent: HtmlElement):
        super().__init__("title", "title", parent=parent)

    @property
    def title_text(self) -> str:
        return self.inner_html

    @title_text.setter
    def title_text(self, value: str) -> None:
        self.inner_html = value

    def render_element(self) -> str:
        indent = '\t' * self.level
        return f"{indent}<{self.open_tag} {self._render_element()}>{self.inner_html}</{self.close_tag}>"


class HtmlHead(HtmlElement):
    def __init__(self, parent: HtmlElement):
        super().__init__("head", "head", parent=parent)
        self._components: Dict[str, Set[HtmlElement]] = {'script': set(), 'meta': set(), 'link': set(), 'style': set()}
        self._title: HtmlTitle = HtmlTitle(parent=self)

    def render_element(self) -> str:
        if not self.has_children:
            return self._render_inline()
        indent = '\t' * self.level
        _header = f'{self.title.render_element()}'
        _scripts = f'{indent}\n'.join(v.render_element() for v in self.scripts)
        _metas = f'{indent}\n'.join(v.render_element() for v in self.metas)
        _styles = f'{indent}\n'.join(v.render_element() for v in self.styles)
        _links = f'{indent}\n'.join(v.render_element() for v in self.links)
        inner_html = f"{indent}\t<meta property=\"{self.inner_html}\"" if len(self.inner_html) else ""
        chdrn = '\n'.join(v for v in (_header, _links, _scripts, _styles, _metas))
        return f"{indent}<{self.open_tag} {self._render_element()}>\n{chdrn}\n{inner_html}{indent}</{self.close_tag}>"

    def _create_element(self, constructor, args: Dict[str, Any] = None):
        element = constructor(self)
        if args:
            for k, v in args.items():
                element.add_attribute(k, v)
        return element

    def add_child(self, node: 'HtmlElement') -> bool:
        if not isinstance(node, HtmlElement):
            return False
        if node.open_tag not in self._components:
            return False
        if not super().add_child(node):
            return False
        self._components[node.open_tag].add(node)
        return True

    def clear_styles(self):
        for value in self.metas:
            self.remove_child(value)
        self._components['style'].clear()

    def clear_metas(self):
        for value in self.metas:
            self.remove_child(value)
        self._components['meta'].clear()

    def clear_links(self):
        for value in self.links:
            self.remove_child(value)
        self._components['link'].clear()

    def clear_scripts(self):
        for value in self.scripts:
            self.remove_child(value)
        self._components['script'].clear()

    @property
    def title(self) -> HtmlTitle:
        return self._title

    @property
    def links(self) -> Generator[HtmlLink, None, None]:
        yield from self._components['link']

    @property
    def metas(self) -> Generator[HtmlMeta, None, None]:
        yield from self._components['meta']

    @property
    def scripts(self) -> Generator[HtmlScript, None, None]:
        yield from self._components['script']

    @property
    def styles(self) -> Generator[HtmlStyle, None, None]:
        yield from self._components['style']

    def append_style(self, args: Dict[str, Any] = None) -> HtmlStyle:
        script = self._create_element(HtmlStyle, args)
        self._components['style'].add(script)
        return script

    def append_script(self, args: Dict[str, Any] = None) -> HtmlScript:
        script = self._create_element(HtmlScript, args)
        self._components['script'].add(script)
        return script

    def append_meta(self, args: Dict[str, Any] = None) -> HtmlMeta:
        meta = self._create_element(HtmlMeta, args)
        self._components['meta'].add(meta)
        return meta

    def append_link(self, args: Dict[str, Any] = None) -> HtmlLink:
        link = self._create_element(HtmlLink, args)
        self._components['link'].add(link)
        return link

    def remove_script(self, value: HtmlScript) -> None:
        if super().remove_child(value):
            self._components['scripts'].remove(value)

    def remove_meta(self, value: HtmlMeta) -> None:
        if super().remove_child(value):
            self._components['meta'].remove(value)

    def remove_link(self, value: HtmlLink) -> None:
        if super().remove_child(value):
            self._components['link'].remove(value)

    def remove_style(self, value: HtmlStyle) -> None:
        if super().remove_child(value):
            self._components['style'].remove(value)
