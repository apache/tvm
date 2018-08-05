"""PyTorch bridge wrapping torch tensor."""

import ctypes
from decorator import decorate
from .. import register_extension, TypeCode

def to_pytorch(module):
    """Wrap a TVM function as PyTorch function

    Parameters
    ----------
    func : Function
        A TVM function that can take positional arguments

    args : Tuple of arguments
        Tuple of arguments to pass to the wrapped function

    Returns
    -------
    func : Function
        A function that can take PyTorch tensor as argument
        in places that used to expect TVM NDArray.
    """
    #import pytorch, check for pytorch tensor
    import torch
    def converter(func, *args):
        args = tuple([FireTensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args])
        return func(*args)

    def wrapper(*args):
        module(*args)

    return decorate(wrapper, converter)

@register_extension
class FireTensor(object):
    """Class to wrap PyTorh tensor"""
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
