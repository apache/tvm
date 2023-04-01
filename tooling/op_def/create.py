"""Tensor creation."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import DType, IntPrimExpr, PrimExpr, Shape, Tensor, Union


@register_op(
    "full",
    te_func="topi.array_api.full",
    sinfo="ShapeBasedCreationSInfo(1, 1, 1)",
)
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
    fill_value = Union(
        PrimExpr,
        Tensor(ndim=0),
        cc_arg2relax=lambda name: f"arg2relax::PrimExprOr0DTensor({name})",
        cc_relax2te=lambda i, name: f'relax2te::PrimExprOr0DTensor(call->args[{i}], "{name}", &_h)',
    )
    dtype = DType(default=None)
    ret = Tensor


@register_op(
    "full_like",
    te_func="topi.array_api.full_like",
    sinfo="ShapeBasedCreationSInfo(0, 1, 0)",
)
class FullLike:
    """TBD"""

    x = Tensor
    fill_value = Union(
        PrimExpr,
        Tensor(ndim=0),
        cc_arg2relax=lambda name: f"arg2relax::PrimExprOr0DTensor({name})",
        cc_relax2te=lambda i, name: f'relax2te::PrimExprOr0DTensor(call->args[{i}], "{name}", &_h)',
    )
    dtype = DType(default=None)
    ret = Tensor


@register_op(
    "ones",
    te_func="topi.array_api.ones",
    sinfo="ShapeBasedCreationSInfo(1, -1, 2)",
)
class Ones:
    """TBD"""

    shape = Shape
    dtype = DType(default=None)
    ret = Tensor


@register_op(
    "ones_like",
    te_func="topi.array_api.ones_like",
    sinfo="ShapeBasedCreationSInfo(0, -1, 0)",
)
class OnesLike:
    """TBD"""

    x = Tensor
    dtype = DType(default=None)
    ret = Tensor


@register_op(
    "zeros",
    te_func="topi.array_api.zeros",
    sinfo="ShapeBasedCreationSInfo(1, -1, 2)",
)
class Zeros:
    """TBD"""

    shape = Shape
    dtype = DType(default=None)
    ret = Tensor


@register_op(
    "zeros_like",
    te_func="topi.array_api.zeros_like",
    sinfo="ShapeBasedCreationSInfo(0, -1, 0)",
)
class ZerosLike:
    """TBD"""

    x = Tensor
    dtype = DType(default=None)
    ret = Tensor


@register_op(
    "triu",
    te_func="topi.array_api.triu",
    sinfo="TrilTriuSInfo",
)
class Triu:
    """TBD"""

    x = Tensor
    k = IntPrimExpr(default=0)
    ret = Tensor


@register_op(
    "tril",
    te_func="topi.array_api.tril",
    sinfo="TrilTriuSInfo",
)
class Tril:
    """TBD"""

    x = Tensor
    k = IntPrimExpr(default=0)
    ret = Tensor
