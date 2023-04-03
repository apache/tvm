"""Tensor creation."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import DType, IntPrimExpr, PrimExpr, Shape, Tensor


@register_op("full")
class Full:
    """TBD

    Attributes
    ----------
    shape
        The shape of the output tensor.
    fill_value
        The value to fill the output tensor with.
    dtype
        The data type of the output tensor.
    ret
        The output tensor.
    """

    shape = Shape
    fill_value = PrimExpr
    dtype = DType(default=None)
    ret = Tensor


@register_op("full_like")
class FullLike:
    """TBD"""

    x = Tensor
    fill_value = PrimExpr
    dtype = DType(default=None)
    ret = Tensor


@register_op("ones")
class Ones:
    """TBD"""

    shape = Shape
    dtype = DType(default=None)
    ret = Tensor


@register_op("ones_like")
class OnesLike:
    """TBD"""

    x = Tensor
    dtype = DType(default=None)
    ret = Tensor


@register_op("zeros")
class Zeros:
    """TBD"""

    shape = Shape
    dtype = DType(default=None)
    ret = Tensor


@register_op("zeros_like")
class ZerosLike:
    """TBD"""

    x = Tensor
    dtype = DType(default=None)
    ret = Tensor


@register_op("triu")
class Triu:
    """TBD"""

    x = Tensor
    diagonal = IntPrimExpr
    ret = Tensor


@register_op("tril")
class Tril:
    """TBD"""

    x = Tensor
    diagonal = IntPrimExpr
    ret = Tensor
