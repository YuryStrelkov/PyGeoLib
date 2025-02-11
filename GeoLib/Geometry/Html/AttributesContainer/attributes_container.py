from typing import Dict, Any, Union, Tuple, Generator


class AttributesContainer:
    def __init__(self):
        self._attributes: Dict[str, Any] = {}

    def __iter__(self) -> Generator[Tuple[str, Any], None, None]:
        yield from self._attributes.items()

    def _add_attribute(self, attribute_key: str, attribute: Union[Any, None]):
        self._attributes.update({attribute_key: attribute})

    def _get_attribute(self, attribute_key: str, constructor: Union[Any, None] = None, *args) -> Union[Any, None]:
        if attribute_key in self._attributes:
            return self._attributes[attribute_key]
        attribute = constructor(*args) if constructor else None
        self._add_attribute(attribute_key, attribute)
        return attribute

    def del_attribute(self, key: str) -> bool:
        if retval := key in self._attributes:
            del self._attributes[key]
        return retval

    def add_attribute(self, key: str, value: Union[Any, None]) -> None:
        self._attributes.update({key: value})

    @property
    def attributes_count(self) -> int:
        return len(self._attributes)

    @property
    def has_attributes(self) -> bool:
        return bool(self.attributes_count)

    @property
    def attributes(self) -> Generator[Tuple[str, Any], None, None]:
        yield from self._attributes.items()

    @property
    def attributes_values(self) -> Generator[Any, None, None]:
        yield from self._attributes.values()

    @property
    def attributes_keys(self) -> Generator[str, None, None]:
        yield from self._attributes.keys()
