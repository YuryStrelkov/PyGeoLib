import dataclasses


@dataclasses.dataclass(frozen=True)
class CssColor:
    red: float
    green: float
    blue: float
    alpha: float

    def __str__(self):
        return f"rgba({self.red}, {self.green}, {self.blue}, {self.alpha})"

    @classmethod
    def rgba(cls, r: float, g: float, b: float, a: float) -> 'CssColor':
        return cls(max(0.0, min(r, 255.0)),
                   max(0.0, min(g, 255.0)),
                   max(0.0, min(b, 255.0)),
                   max(0.0, min(a, 1.0)))

    @staticmethod
    def rgb(r: float, g: float, b: float) -> 'CssColor':
        return CssColor.rgba(r, g, b, 1.0)

    @staticmethod
    def gray(r: float) -> 'CssColor':
        return CssColor.rgba(r, r, r, 1.0)
