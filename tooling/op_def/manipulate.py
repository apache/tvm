"""Shape manipulation operators."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import (
    Array,
    Axes,
    Axis,
    Float,
    IndexMap,
    Int,
    IntPrimExpr,
    Optional,
    Shape,
    Tensor,
    Union,
)

# ##### Concat/Split #####


@register_op(
    "concat",
    te_func="topi.array_api.concat",
    sinfo_fallback="ConcatSInfoFallback",
)
class Concat:
    """TBD"""

    x = Array(Tensor)
    axis = Axis(of="x", default=0)
    ret = Tensor


@register_op(
    "split",
    te_func="topi.array_api.split",
    sinfo_fallback="SplitSInfoFallback",
)
class Split:
    """TBD"""

    x = Tensor
    indices_or_sections = Union(
        Int,
        Array(IntPrimExpr, restrict=True),
        cc_arg2relax=lambda name: f"AttrExpr({name})",
        cc_relax2te=lambda i, name: f"ObjectFromOpaque()(call->args[{i}])",
    )
    axis = Axis(of="x", default=0)
    ret = Array(Tensor)


# ##### Shape Manipulation #####


@register_op(
    "reshape",
    te_func="topi.array_api.reshape",
    sinfo_fallback="ReshapeSInfoFallback",
)
class Reshape:
    """TBD"""

    x = Tensor
    shape = Shape
    ret = Tensor


@register_op(
    "flatten",
    te_func="topi.array_api.flatten",
    sinfo_fallback="FlattenSInfoFallback",
)
class Flatten:
    """TBD"""

    x = Tensor
    start_dim = Axis(of="x", default=0)
    end_dim = Axis(of="x", default=-1)
    ret = Tensor


@register_op(
    "layout_transform",
    sinfo="LayoutTransformSInfo",
)
class LayoutTransform:
    """TBD"""

    x = Tensor
    index_map = IndexMap
    pad_value = Optional(Float, default=None)
    ret = Tensor


##### Axes Manipulation #####


@register_op(
    "expand_dims",
    te_func="topi.array_api.expand_dims",
    sinfo_fallback="ExpandDimsSInfoFallback",
)
class ExpandDims:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", is_insertion=True, normalize=False)
    ret = Tensor


@register_op(
    "permute_dims",
    te_func="topi.array_api.permute_dims",
    sinfo_fallback="PermuteDimsSInfoFallback",
)
class PermuteDims:
    """TBD

    TODO(@junrushao): In numpy it is called transpose
    """

    x = Tensor
    axes = Optional(Axes(of="x", normalize=False), default=None)
    ret = Tensor


@register_op(
    "squeeze",
    te_func="topi.array_api.squeeze",
    sinfo_fallback="SqueezeSInfoFallback",
)
class Squeeze:
    """TBD"""

    x = Tensor(allow_ndim_only=False)
    axis = Optional(Axes(of="x"), default=None)
    ret = Tensor


##### Repetitive Copy #####


@register_op(
    "broadcast_to",
    te_func="topi.array_api.broadcast_to",
    sinfo="BroadcastToSInfo",
)
class BroadcastTo:
    """TBD"""

    x = Tensor
    shape = Shape
    ret = Tensor


@register_op(
    "repeat",
    te_func="topi.array_api.repeat",
    sinfo_fallback="RepeatSInfoFallback",
)
class Repeat:
    """TBD"""

    x = Tensor
    repeats = Array(
        IntPrimExpr,
        cc_arg2relax=lambda name: f"AttrExpr({name})",
        cc_relax2te=lambda i, name: f"ArrayFromOpaque()(call->args[{i}])",
    )
    axis = Optional(Axis(of="x"), default=None)
    ret = Tensor


@register_op(
    "tile",
    te_func="topi.array_api.tile",
    sinfo_fallback="TileSInfoFallback",
)
class Tile:
    """TBD"""

    x = Tensor
    repeats = Shape
    ret = Tensor
