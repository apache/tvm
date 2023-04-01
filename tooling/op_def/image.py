"""Image-related operators"""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import Array, DType, Float, Shape, Str, Tensor


@register_op(
    "image.resize2d",
    te_func="topi.array_api.image.resize2d",
    sinfo_fallback="Resize2dSInfoFallback",
    sinfo_out_dtype="out_dtype",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutResize2d"),
        ("TMixedPrecisionPolicy", "MixedPrecisionPolicyKind::kFollow"),
    ],
)
class Resize2d:
    """TBD"""

    x = Tensor
    size = Shape
    roi = Array(Float, length=4, default=(0.0, 0.0, 0.0, 0.0))
    layout = Str(default="NCHW")
    method = Str(default="linear")
    coordinate_transformation_mode = Str(default="half_pixel")
    rounding_method = Str(default="round")
    bicubic_alpha = Float(default=-0.5)
    bicubic_exclude = Float(default=0.0)
    extrapolation_value = Float(default=0.0)
    out_dtype = DType(default=None)
    ret = Tensor
