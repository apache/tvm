"""Linear algebra operations."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import DType, Tensor


@register_op(
    "matmul",
    te_func="topi.array_api.matmul",
    sinfo_fallback="MatmulSInfoFallback",
    sinfo_out_dtype="out_dtype",
    attrs=[
        ("TMixedPrecisionPolicy", "MixedPrecisionPolicyKind::kAlways"),
        ("FInferMixedPrecision", "InferMixedPrecisionMatmul"),
    ],
)
class Matmul:
    """TBD"""

    x1 = Tensor(allow_ndim_only=False)
    x2 = Tensor(allow_ndim_only=False)
    out_dtype = DType(default=None)
    ret = Tensor
