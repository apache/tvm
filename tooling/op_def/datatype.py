"""Data type conversion operators."""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import DType, Tensor


@register_op("astype")
class AsType:
    """TBD"""

    x = Tensor
    dtype = DType
    ret = Tensor
