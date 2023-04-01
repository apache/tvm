"""VM-related operators"""
from ..registry import register_op
from ..ty import AnyRelaxExpr, DType, ExternFunc, IntPrimExpr, Shape, Tensor, TupleExpr

# pylint: disable=too-few-public-methods


@register_op("vm.alloc_storage", sinfo="InferStructInfoReturnsObject")
class AllocStorage:
    """Allocate a storage with specific size and dtype on a specific device.
    The allocated storage can be used to create tensors in-place.
    The storage is automatically managed by the VM.

    Attributes
    ----------
    size
        The shape of the storage.
    runtime_device_index
        The index of the device on which the storage is allocated.
    dtype
        The data type of the storage.
    ret
        The allocated storage.
    """

    size = Shape
    runtime_device_index = IntPrimExpr
    dtype = DType
    ret = AnyRelaxExpr


@register_op("vm.alloc_tensor", sinfo="InferStructInfoVMAllocTensor")
class AllocTensor:
    """Allocate a tensor with specific shape, dtype on a specific device at the specific offset
    on a storage created by R.vm.alloc_storage. The tensor is automatically managed by the VM.

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


@register_op("vm.call_tir_dyn", sinfo="InferStructInfoReturnsVoid")
class CallTIRDyn:
    """Call a TIR function with dynamic arguments.

    Attributes
    ----------
    func
        The TIR function to be called.
    args
        The arguments to the TIR function.
    ret
        The call node created
    """

    func = ExternFunc
    args = TupleExpr
    ret = Tensor
