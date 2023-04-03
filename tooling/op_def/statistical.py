"""Statistical operators."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Axes, Bool, Tensor


@register_op("max")
class Max:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    keepdims = Bool
    ret = Tensor


@register_op("min")
class Min:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    keepdims = Bool
    ret = Tensor


@register_op("sum")
class Sum:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    keepdims = Bool
    ret = Tensor


@register_op("prod")
class Prod:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    keepdims = Bool
    ret = Tensor


@register_op("mean")
class Mean:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    keepdims = Bool
    ret = Tensor


@register_op("std")
class Std:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    keepdims = Bool
    ret = Tensor


@register_op("variance")
class Variance:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    keepdims = Bool
    ret = Tensor
