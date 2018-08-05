"""PyTorch bridge wrapping torch tensor."""

from .. import register_extension, TypeCode
import ctypes
import numpy as np
from decorator import decorate

def to_pytorch(module):
    #import pytorch, check for pytorch tensor
    import torch
    def converter(func, *args):
        new_args = tuple([FireTensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args])
        return func(*new_args)

    def wrapper(*args):
        module(*args)

    return decorate(wrapper, converter)

@register_extension
class FireTensor(object):
    _tvm_tcode = TypeCode.ARRAY_HANDLE

    def __init__(self, tensor):
        import torch
        self.handle = torch._C._to_dlpack(tensor)
        self.name = self.get_name()

    def get_name(self):
        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        return ctypes.pythonapi.PyCapsule_GetName(self.handle)

    def to_torch(self):
        return torch._C._from_dlpack(self.handle)

    @property
    def _tvm_handle(self):
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        return ctypes.pythonapi.PyCapsule_GetPointer(self.handle, self.name)
