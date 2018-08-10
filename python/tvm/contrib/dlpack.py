"""Wrapping functions to bridge frameworks with DLPack support to TVM"""
from decorator import decorate

from .. import ndarray

def convert_func(tvm_func, tensor_type, to_dlpack_func):
    """Convert a tvm function into one that accepts a tensor from another
       framework, provided the other framework supports DLPACK

    Parameters
    ----------
    tvm_func: Built tvm function operating on arrays

    tensor_type: Type of the tensors of the target framework

    to_dlpack_func: Function to convert the source tensors to DLPACK
    """
    def _wrapper(*args):
        args = tuple(ndarray.from_dlpack(to_dlpack_func(arg))\
            if isinstance(arg, tensor_type) else arg for arg in args)
        return tvm_func(*args)
    return _wrapper

def to_pytorch(tvm_func):
    """Convert a tvm function into one that accepts PyTorch tensors

    Parameters
    ----------
    tvm_func: Built tvm function operating on arrays

    Returns
    -------
    wrapped_func: Wrapped tvm function that operates on PyTorch tensors
    """
    import torch
    import torch.utils.dlpack
    return convert_func(tvm_func, torch.Tensor, torch.utils.dlpack.to_dlpack)
