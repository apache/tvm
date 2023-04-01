"""Shape manipulation operators."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import (
    Array,
    Axes,
    Axis,
    Float,
    IndexMap,
    IntPrimExpr,
    Optional,
    Shape,
    Tensor,
)

##### Concat/Split #####


@register_op("concat")
class Concat:
    """TBD"""

    x = Array(Tensor)
    axis = Axis(of="x", default=0)
    ret = Tensor


@register_op("split")
class Split:
    """TBD"""

    x = Tensor
    indices_or_sections = IntPrimExpr
    axis = Axis(of="x", default=0)
    ret = Array(Tensor)


##### Shape Manipulation #####


@register_op("reshape")
class Reshape:
    """TBD"""

    x = Tensor
    shape = Shape
    ret = Tensor


@register_op("flatten")
class Flatten:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    ret = Tensor


@register_op("layout_transform")
class LayoutTransform:
    """TBD"""

    x = Tensor
    index_map = IndexMap
    pad_value = Optional(Float, default=None)
    ret = Tensor


##### Axes Manipulation #####


@register_op("expand_dims")
class ExpandDims:
    """TBD"""

    x = Tensor
    axis = Axes(of="x", is_insertion=True)
    ret = Tensor


@register_op("squeeze")
class Squeeze:
    """TBD"""

    x = Tensor
    axis = Axes(of="x")
    ret = Tensor


@register_op("permute_dims")
class PermuteDims:
    """TBD"""

    x = Tensor
    axes = Optional(Axes(of="x", normalize=False), default=None)
    ret = Tensor


##### Repetitive Copy #####


@register_op("broadcast_to")
class BroadcastTo:
    """TBD"""

    x = Tensor
    shape = Shape
    ret = Tensor


@register_op("repeat")
class Repeat:
    """TBD"""

    x = Tensor
    repeats = IntPrimExpr
    axis = Axis(of="x", default=0)
    ret = Tensor


@register_op("tile")
class Tile:
    """TBD"""

    x = Tensor
    repeats = Array(IntPrimExpr)
    ret = Tensor
