"""Builtin methods in Relax"""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import DType, Int, Shape, Tensor


@register_op("builtin.alloc_tensor", sinfo="InferStructInfoAllocTensor")
class AllocTensor:
    """Construct a Call to allocate a tensor with specific shape, dtype, and the index of
    the device it is constructed on.

    Attributes
    ----------
    shape
        The shape of the tensor.
    dtype
        The data type of the tensor.
    runtime_device_index
        The index of the device it is constructed on.
    ret
        The created call node.
    """

    shape = Shape
    dtype = DType
    runtime_device_index = Int
    ret = Tensor


@register_op("builtin.stop_lift_params", sinfo="InferStructInfoIdentical")
class StopLiftParams:
    """An indicator op that the consumers of input tensor should not be
    lifted to transform_params function.

    Attributes
    ----------
    x
        The input tensor.
    ret
        The created call node.
    """

    x = Tensor
    ret = Tensor
