"""Set operations."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Axis, BoolPrimExpr, Optional, Tensor


@register_op("unique")
class Unique:
    """TBD"""

    x = Tensor
    sorted = BoolPrimExpr(default=True)
    return_index = BoolPrimExpr(default=False)
    return_inverse = BoolPrimExpr(default=False)
    return_counts = BoolPrimExpr(default=False)
    axis = Optional(Axis(of="x"), default=None)
    ret = Tensor  # TODO: fix the return type
