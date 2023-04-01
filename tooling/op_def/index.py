"""Indexing operations."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Array, Axes, Axis, IntPrimExpr, IntTensor, Optional, Tensor


# @register_op("take")
class Take:
    """TBD"""

    x = Tensor
    indices = IntTensor(ndim=1)
    axis = Optional(Axis(of="x"), default=None)
    ret = Tensor


# @register_op("strided_slice")
class StridedSlice:
    """TBD"""

    x = Tensor
    axes = Axes(of="x")
    begin = Array(IntPrimExpr)
    end = Array(IntPrimExpr)
    strides = Optional(
        Array(IntPrimExpr),
        default=None,
        cc_arg2relax=lambda name: f"AttrExpr({name})",
    )
    ret = Tensor
