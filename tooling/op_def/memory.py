"""Relax memory allocation"""
# pylint: disable=too-few-public-methods
from ..registry import register_op
from ..ty import AnyRelaxExpr, DType, IntPrimExpr, Shape, Str, Tensor


@register_op("memory.alloc_storage", sinfo="InferStructInfoReturnsObject")
class AllocStorage:
    """Allocate a chunk of memory storage with specific size, dtype on a specific device
    on its specific storage scope. The allocated storage can be used to create tensors in-place.
    The storage will only be freed when the program exits or when the storage is killed by
    R.memory.kill_storage.

    Attributes
    ----------
    size
        The shape of the storage.
    virtual_device_index
        The index of the device on which the storage is allocated.
    storage_scope
        The storage scope of the storage.
    dtype
        The data type of the storage.
    ret
        The allocated storage.
    """

    size = Shape
    virtual_device_index = IntPrimExpr
    storage_scope = Str
    dtype = DType
    ret = AnyRelaxExpr


@register_op("memory.alloc_tensor", sinfo="InferStructInfoMemoryAllocTensor")
class AllocTensor:
    """Allocate a tensor with specific shape, dtype on a specific device at the specific offset
    on a storage created by R.memory.alloc_storage.
    The tensor will only be freed when the program exits or when the tensor is killed by
    R.memory.kill_tensor.

    Attributes
    ----------
    storage
        The storage on which the tensor is allocated.
    offset
        The offset of the tensor on the storage.
    shape
        The shape of the tensor.
    dtype
        The data type of the tensor.
    ret
        The allocated tensor.
    """

    storage = AnyRelaxExpr
    offset = IntPrimExpr
    shape = Shape
    dtype = DType
    ret = Tensor


@register_op("memory.kill_storage", sinfo="InferStructInfoReturnsVoid")
class KillStorage:
    """Kill a storage created by R.memory.alloc_storage.

    Attributes
    ----------
    storage
        The storage being allocated by R.memory.alloc_storage.
    ret
        The call node created.
    """

    storage = AnyRelaxExpr
    ret = AnyRelaxExpr


@register_op("memory.kill_tensor", sinfo="InferStructInfoReturnsVoid")
class KillTensor:
    """Kill a tensor created by R.memory.alloc_tensor.

    Attributes
    ----------
    tensor
        The tensor being allocated by R.memory.alloc_tensor.
    ret
        The call node created.
    """

    tensor = Tensor
    ret = AnyRelaxExpr
