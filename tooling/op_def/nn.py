"""Neural network operators."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import (
    Array,
    Axes,
    Axis,
    Bool,
    DType,
    Float,
    Int,
    Optional,
    Shape,
    Str,
    Tensor,
)


@register_op("nn.conv2d")
class Conv2d:
    """TBD"""

    data = Tensor
    weight = Tensor
    strides = Array(Int, length=2, default=(1, 1))
    padding = Array(Int, length=2, default=(0, 0))
    dilation = Array(Int, length=2, default=(1, 1))
    groups = Int(default=1)
    data_layout = Str(default="NCHW")
    kernel_layout = Str(default="OIHW")
    out_layout = Str(default="")
    out_dtype = DType(default=None)
    ret = Tensor


@register_op("nn.conv2d_transpose")
class Conv2dTranspose:
    """TBD"""

    data = Tensor
    weight = Tensor
    strides = Array(Int, length=2, default=(1, 1))
    padding = Array(Int, length=2, default=(0, 0))
    dilation = Array(Int, length=2, default=(1, 1))
    groups = Int(default=1)
    data_layout = Str(default="NCHW")
    kernel_layout = Str(default="OIHW")
    out_layout = Str(default="")
    out_dtype = DType(default=None)
    ret = Tensor


@register_op("nn.max_pool2d")
class MaxPool2d:
    """TBD"""

    data = Tensor
    pool_size = Array(Int, length=2, default=(1, 1))
    strides = Array(Int, length=2, default=(1, 1))
    padding = Array(Int, length=2, default=(0, 0))
    dilation = Array(Int, length=2, default=(1, 1))
    ceil_mode = Bool(default=False)
    layout = Str(default="NCHW")
    out_layout = Str(default="")
    ret = Tensor


@register_op("nn.avg_pool2d")
class AvgPool2d:
    """TBD"""

    data = Tensor
    pool_size = Array(Int, length=2, default=(1, 1))
    strides = Array(Int, length=2, default=(1, 1))
    padding = Array(Int, length=2, default=(0, 0))
    dilation = Array(Int, length=2, default=(1, 1))
    ceil_mode = Bool(default=False)
    layout = Str(default="NCHW")
    out_layout = Str(default="")
    ret = Tensor


@register_op("nn.adaptive_max_pool2d")
class AdaptiveAvgPool2d:
    """TBD"""

    data = Tensor
    output_size = Optional(
        Array(Int),
        default=None,
        cc_arg2relax=lambda name: f"AttrExpr({name})",
    )
    layout = Str(default="NCHW")
    out_layout = Str(default="")
    ret = Tensor


@register_op("nn.relu")
class ReLU:
    """TBD"""

    data = Tensor
    ret = Tensor


@register_op("nn.gelu")
class GeLU:
    """TBD"""

    data = Tensor
    ret = Tensor


@register_op("nn.silu")
class SiLU:
    """TBD"""

    data = Tensor
    ret = Tensor


@register_op("nn.softmax")
class Softmax:
    """TBD"""

    data = Tensor
    axis = Axis(of="data", default=-1)
    ret = Tensor


@register_op("nn.log_softmax")
class LogSoftmax:
    """TBD"""

    data = Tensor
    axis = Axis(of="data", default=-1)
    ret = Tensor


@register_op("nn.batch_norm")
class BatchNorm:
    """TBD"""

    data = Tensor
    gamma = Tensor
    beta = Tensor
    moving_mean = Tensor
    moving_var = Tensor
    axes = Axes(of="data")
    epsilon = Float(default=1e-05)
    center = Bool(default=True)
    scale = Bool(default=True)
    ret = Tensor


@register_op("nn.layer_norm")
class LayerNorm:
    """TBD"""

    data = Tensor
    gamma = Tensor
    beta = Tensor
    axes = Axes(of="data")
    epsilon = Float(default=1e-05)
    center = Bool(default=True)
    scale = Bool(default=True)
    ret = Tensor


@register_op("nn.group_norm")
class GroupNorm:
    """TBD"""

    data = Tensor
    gamma = Tensor
    beta = Tensor
    num_groups = Int
    channel_axis = Axis(of="data")
    epsilon = Float(default=1e-05)
    center = Bool(default=True)
    scale = Bool(default=True)
    ret = Tensor


@register_op("nn.dropout")
class Dropout:
    """TBD"""

    data = Tensor
    rate = Float(default=0.5)
    ret = Tensor


@register_op("nn.attention")
class Attention:
    """TBD"""

    query = Tensor
    key = Tensor
    value = Tensor
    ret = Tensor


@register_op("nn.cross_entropy_with_logits")
class CrossEntropyWithLogits:
    """TBD"""

    predictions = Tensor
    labels = Tensor
    ret = Tensor
