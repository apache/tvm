"""Linear algebra operations."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import DType, Optional, Tensor


@register_op("matmul")
class Matmul:
    """TBD"""

    x1 = Tensor
    x2 = Tensor
    out_dtype = DType
    ret = Tensor
