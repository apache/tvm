"""Statistical operators."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Axes, Bool, Tensor


@register_op(
    "max",
    te_func="topi.array_api.max",
    sinfo_fallback="ReduceSInfoFallback(DataType::Void())",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutStatistical"),
    ],
)
class Max:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", default=None)
    keepdims = Bool(default=False)
    ret = Tensor


@register_op(
    "min",
    te_func="topi.array_api.min",
    sinfo_fallback="ReduceSInfoFallback(DataType::Void())",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutStatistical"),
    ],
)
class Min:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", default=None)
    keepdims = Bool(default=False)
    ret = Tensor


@register_op(
    "sum",
    te_func="topi.array_api.sum",
    sinfo_fallback="ReduceSInfoFallback(DataType::Void())",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutStatistical"),
    ],
)
class Sum:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", default=None)
    keepdims = Bool(default=False)
    ret = Tensor


@register_op(
    "prod",
    te_func="topi.array_api.prod",
    sinfo_fallback="ReduceSInfoFallback(DataType::Void())",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutStatistical"),
    ],
)
class Prod:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", default=None)
    keepdims = Bool(default=False)
    ret = Tensor


@register_op(
    "mean",
    te_func="topi.array_api.mean",
    sinfo_fallback="ReduceSInfoFallback(DataType::Void())",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutStatistical"),
    ],
)
class Mean:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", default=None)
    keepdims = Bool(default=False)
    ret = Tensor


@register_op(
    "std",
    te_func="topi.array_api.std",
    sinfo_fallback="ReduceSInfoFallback(DataType::Void())",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutStatistical"),
    ],
)
class Std:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", default=None)
    keepdims = Bool(default=False)
    ret = Tensor


@register_op(
    "variance",
    te_func="topi.array_api.variance",
    sinfo_fallback="ReduceSInfoFallback(DataType::Void())",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutStatistical"),
    ],
)
class Variance:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", default=None)
    keepdims = Bool(default=False)
    ret = Tensor
