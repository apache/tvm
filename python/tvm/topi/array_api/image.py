from typing import Sequence

from tvm._ffi import register_func
from tvm.ir import PrimExpr
from tvm.te import Tensor


@register_func("topi.array_api.image.resize2d")
def resize2d(  # pylint: disable=invalid-name,too-many-arguments
    x: Tensor,
    size: Sequence[PrimExpr],
    roi: Sequence[float],  # length=4
    layout: str = "NCHW",
    method: str = "linear",
    coordinate_transformation_mode: str = "half_pixel",
    rounding_method: str = "round",
    bicubic_alpha: float = -0.5,
    bicubic_exclude: float = 0.0,
    extrapolation_value: float = 0.0,
    out_dtype: str = None,
):  # pylint: disable=import-outside-toplevel
    """TBD"""
    from ..image import resize2d as legacy_resize2d

    # pylint: enable=import-outside-toplevel
    return legacy_resize2d(
        data=x,
        roi=roi,
        size=size,
        layout=layout,
        method=method,
        coordinate_transformation_mode=coordinate_transformation_mode,
        rounding_method=rounding_method,
        bicubic_alpha=bicubic_alpha,
        bicubic_exclude=bicubic_exclude,
        extrapolation_value=extrapolation_value,
        out_dtype=out_dtype,
    )
