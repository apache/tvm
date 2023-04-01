"""Data type conversion operators."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import DType, Tensor


@register_op(
    "astype",
    te_func="topi.array_api.astype",
    sinfo_fallback="AsTypeSInfoFallback",
    sinfo_out_dtype="dtype",
    legalization="AsTypeLegalize",
    attrs=[
        ("FRelaxInferLayout", "InferLayoutUnaryEwise"),
        ("TMixedPrecisionPolicy", "MixedPrecisionPolicyKind::kFollow"),
    ],
)
class AsType:
    """TBD"""

    x = Tensor
    dtype = DType
    ret = Tensor
