"""Set operations."""
# pylint: disable=too-few-public-methods
from ..ty import Axis, BoolPrimExpr, Optional, Tensor


class Unique:
    """TODO(@junrushao): incorporate this operator to op schema. Right now this is just too messy"""

    x = Tensor
    sorted = BoolPrimExpr(default=True)
    return_index = BoolPrimExpr(default=False)
    return_inverse = BoolPrimExpr(default=False)
    return_counts = BoolPrimExpr(default=False)
    axis = Optional(Axis(of="x"), default=None)
    ret = Tensor  # TODO: fix the return type
