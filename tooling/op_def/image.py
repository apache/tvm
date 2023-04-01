"""Image-related operators"""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Array, DType, Float, Shape, Str, Tensor


@register_op("image.resize2d")
class Resize2d:
    """TBD"""

    x = Tensor
    size = Shape
    roi = Array(Float, length=4, default=(0.0, 0.0, 0.0, 0.0))
    layout = Str(default="NCHW")
    method = Str(default="linear")
    coordinate_transformation_mode = Str(default="half_pixel")
    rounding_method = Str(default="round")
    cubic_alpha = Float(default=-0.5)
    extrapolation_value = Float(default=0.0)
    out_dtype = DType(default=None)
    ret = Tensor
