from typing import Generator
import dataclasses
from .css_constatnts import CssUnits


@dataclasses.dataclass(frozen=True)
class CssUnit:
    value: float
    unit: CssUnits

    def __iter__(self) -> Generator['CssUnit', None, None]:
        yield self

    def __str__(self) -> str:
        if self.unit == CssUnits.AUTO:
            return f"{self.unit}"
        return f"{self.value}{self.unit}"

    @classmethod
    def create(cls, value: float, unit: CssUnits) -> 'CssUnit':
        try:
            _value = float(value)
            match unit:
                case CssUnits.CM:
                    return cls(_value, CssUnits.CM)
                case CssUnits.INCH:
                    return cls(_value, CssUnits.INCH)
                case CssUnits.PIX:
                    return cls(_value, CssUnits.PIX)
                case CssUnits.PERCENT:
                    return cls(_value, CssUnits.PERCENT)
                case CssUnits.REM:
                    return cls(_value, CssUnits.REM)
                case CssUnits.EM:
                    return cls(_value, CssUnits.EM)
                case CssUnits.AUTO:
                    return cls(_value, CssUnits.AUTO)
                case _:
                    return cls(_value, CssUnits.NONE)
        except ValueError as _:
            # todo log error...
            return cls(0.0, CssUnits.NONE)

    @staticmethod
    def centimeter(value: float) -> 'CssUnit':
        return CssUnit.create(value, CssUnits.CM)

    @staticmethod
    def inch(value: float) -> 'CssUnit':
        return CssUnit.create(value, CssUnits.INCH)

    @staticmethod
    def pixel(value: float) -> 'CssUnit':
        return CssUnit.create(value, CssUnits.INCH)

    @staticmethod
    def percent(value: float) -> 'CssUnit':
        return CssUnit.create(value, CssUnits.PERCENT)

    @staticmethod
    def none(value: float) -> 'CssUnit':
        return CssUnit.create(value, CssUnits.NONE)

    @staticmethod
    def auto() -> 'CssUnit':
        return CssUnit.create(0.0, CssUnits.AUTO)

    @staticmethod
    def root_ephemeral_unit(value: float) -> 'CssUnit':
        return CssUnit.create(value, CssUnits.REM)

    @staticmethod
    def ephemeral_unit(value: float) -> 'CssUnit':
        return CssUnit.create(value, CssUnits.EM)
