"""Search operations."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Axes, Bool, BoolTensor, Tensor


@register_op(
    "where",
    te_func="topi.array_api.where",
    sinfo="WhereSInfo",
)
class Where:
    """TBD"""

    condition = BoolTensor
    x = Tensor
    y = Tensor
    ret = Tensor


@register_op(
    "argmax",
    te_func="topi.array_api.argmax",
    sinfo_fallback="ReduceSInfoFallback(DataType::Int(64))",
)
class ArgMax:
    """TBD"""

    a = Tensor
    axis = Axes(of="a", default=None)
    keepdims = Bool(default=False)
    ret = Tensor


@register_op(
    "argmin",
    te_func="topi.array_api.argmin",
    sinfo_fallback="ReduceSInfoFallback(DataType::Int(64))",
)
class ArgMin:
    """TBD"""

    a = Tensor
    axis = Axes(of="a", default=None)
    keepdims = Bool(default=False)
    ret = Tensor
