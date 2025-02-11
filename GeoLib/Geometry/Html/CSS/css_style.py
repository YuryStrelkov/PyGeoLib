from .css_color import CssColor
from .css_units import CssUnit
from .css_constatnts import CssGlobal, CssPosition, CssTextAlign, CssVisibility, CssUnits
from typing import Union, Tuple, Iterable, Any
from ..AttributesContainer import AttributesContainer


class CssElementStyle(AttributesContainer):
    def _set_var(self, key, value: Union[int, float, CssUnit]) -> None:
        if isinstance(value, (int, float)):
            self._add_attribute(key, CssUnit(float(value), CssUnits.PIX))
            return
        if isinstance(value, CssUnit):
            self._add_attribute(key, value)
            return
        raise ValueError(f"Unsupported {key} type: \"{type(value)}\"")

    def _set_vars(self, key: str, max_args_length: int, value: Union[CssGlobal, Tuple[CssUnit, ...]]):
        if isinstance(value, CssGlobal):
            self._add_attribute(key, value)
            return
        if isinstance(value, Tuple) and all(isinstance(v, CssUnit) for v in value):
            if len(value) > max_args_length:
                self._add_attribute(key, value[:max_args_length])
            else:
                self._add_attribute(key, value)
            return
        raise ValueError(f"Unsupported {key} type: {type(value)}...")

    def _set_color_var(self, key: str, value: Union[str, int, CssColor]):
        if isinstance(value, int):
            self._add_attribute(key, CssColor(float(value & 0xff),
                                              float((value & (0xff << 8 ) >> 8 )),
                                              float((value & (0xff << 16) >> 16)),
                                              float((value & (0xff << 24) >> 24)) / 255.0))
            return
        if isinstance(value, (str, CssColor)):
            self._add_attribute(key, value)
            return
        raise ValueError(f"Unsupported {key} type: {type(value)}...")

    @property
    def position(self) -> Union[CssPosition, CssGlobal]:
        return self._get_attribute("position", lambda _: CssGlobal.UNSET)

    @position.setter
    def position(self, value: Union[CssPosition, CssGlobal]) -> None:
        if isinstance(value, (CssPosition, CssGlobal)):
            self._add_attribute("position", value)

    @property
    def text_align(self) -> Union[CssTextAlign, CssGlobal]:
        return self._get_attribute("text-align", lambda _: CssGlobal.UNSET)

    @text_align.setter
    def text_align(self, value: CssTextAlign) -> None:
        if isinstance(value, (CssTextAlign, CssGlobal)):
            self._add_attribute("text-align", value)

    @property
    def visibility(self) -> Union[CssPosition, CssGlobal]:
        return self._get_attribute("visibility", lambda _: CssGlobal.UNSET)

    @visibility.setter
    def visibility(self, value: Union[CssPosition, CssGlobal]) -> None:
        if isinstance(value, (CssVisibility, CssGlobal)):
            self._add_attribute("visibility", value)

    @property
    def width(self) -> CssUnit:
        return self._get_attribute("width", CssUnit.pixel, 0.0)

    @property
    def height(self) -> CssUnit:
        return self._get_attribute("height", CssUnit.pixel, 0.0)

    @width.setter
    def width(self, value: Union[int, float, CssUnit]) -> None:
        self._set_var("width", value)

    @height.setter
    def height(self, value: CssUnit) -> None:
        self._set_var("height", value)

    @property
    def top(self) -> CssUnit:
        return self._get_attribute("top", CssUnit.pixel, 0.0)

    @property
    def left(self) -> CssUnit:
        return self._get_attribute("left", CssUnit.pixel, 0.0)

    @top.setter
    def top(self, value: CssUnit) -> None:
        self._set_var("top", value)

    @left.setter
    def left(self, value: CssUnit) -> None:
        self._set_var("left", value)

    @property
    def padding(self) -> Union[CssGlobal, Tuple[CssUnit, ...]]:
        return self._get_attribute("padding", lambda _: CssGlobal.UNSET)

    @padding.setter
    def padding(self, value: Union[CssGlobal, Tuple[CssUnit, ...]]) -> None:
        self._set_vars("padding", 4, value)

    @property
    def margin(self) -> Union[CssGlobal, Tuple[CssUnit, ...]]:
        return self._get_attribute("margin", lambda _: CssGlobal.UNSET)

    @margin.setter
    def margin(self, value: Union[CssGlobal, Tuple[CssUnit, ...]]) -> None:
        self._set_vars("margin", 4, value)

    @property
    def padding_block(self) -> Union[CssGlobal, Tuple[CssUnit, ...]]:
        return self._get_attribute("padding", lambda _: CssGlobal.UNSET)

    @padding_block.setter
    def padding_block(self, value: Union[CssGlobal, Tuple[CssUnit, ...]]) -> None:
        self._set_vars("padding-block", 2, value)

    @property
    def border_radius(self) -> Union[CssGlobal, Tuple[CssUnit, ...]]:
        return self._get_attribute("padding", lambda _: CssGlobal.UNSET)

    @border_radius.setter
    def border_radius(self, value: Union[CssGlobal, Tuple[CssUnit, ...]]) -> None:
        self._set_vars("border-radius", 4, value)

    @property
    def z_index(self) -> int:
        return self._get_attribute('z-index', int)

    @z_index.setter
    def z_index(self, value: int) -> None:
        self._add_attribute("z-index", int(value))

    @property
    def color(self) ->  Union[CssColor, str]:
        return self._get_attribute('color', CssColor.gray, 255)

    @color.setter
    def color(self, value: Union[str, int, CssColor]) -> None:
        self._set_color_var('color', value)

    @property
    def background_color(self) -> Union[CssColor, str]:
        return self._get_attribute('background-color', CssColor.gray, 255)

    @background_color.setter
    def background_color(self, value: Union[str, int, CssColor]) -> None:
        self._set_color_var('background-color', value)

    def __str__(self) -> str:
        def _parce_item(item: Union[Iterable, Any]) -> str:
            return ' '.join(str(vi) for vi in item )
        return '; '.join(f"{k}: {_parce_item(v)}"for k, v in self.attributes)

