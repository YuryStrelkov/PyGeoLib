from enum import Enum


class CssEnum(Enum):
    def __str__(self) -> str:
        return self.value


class CssUnits(CssEnum):
    INCH = 'in'
    CM = 'cm'
    PIX = 'px'
    EM = 'em'
    REM = 'rem'
    PERCENT = '%'
    NONE = ''
    AUTO = 'auto'


class CssPosition(CssEnum):
    STATIC = "static"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    FIXED = "fixed"
    STICKY = "sticky"


class CssTextAlign(CssEnum):
    # / *Keyword values * /
    START = "start"
    END = "end"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    JUSTIFY = "justify"
    MATCH_PARENT = "match - parent"

    # / *Block alignment values(Non - standard syntax) * /
    MOZ_CENTER = "-moz - center"
    WEBKIT_CENTER = "-webkit - center"


class CssVisibility(CssEnum):
    # / *Keyword values * /
    VISIBLE = "visible"
    HIDDEN = "hidden"
    COLLAPSE = "collapse"


class CssGlobal(CssEnum):
    # / *Global values * /
    INHERIT = "inherit"
    INITIAL = "initial"
    REVERT = "revert"
    REVERT_LAYER = "revert - layer"
    UNSET = "unset"
