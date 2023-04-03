"""Search operations."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Axes, Bool, Optional, Tensor


@register_op("where")
class Where:
    """TBD"""

    condition = Tensor
    x = Tensor
    y = Tensor
    ret = Tensor


@register_op("argmax")
class ArgMax:
    """TBD"""

    a = Tensor
    axis = Optional(Axes(of="a"), default=None)
    keepdims = Optional(Bool, default=False)
    ret = Tensor


@register_op("argmin")
class ArgMin:
    """TBD"""

    a = Tensor
    axis = Optional(Axes(of="a"), default=None)
    keepdims = Optional(Bool, default=False)
    ret = Tensor
