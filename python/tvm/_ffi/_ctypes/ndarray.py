"""Runtime NDArray api"""
from __future__ import absolute_import

import ctypes
from ..base import _LIB, check_call
from ..runtime_ctypes import TVMArrayHandle
from .types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func, _return_handle

class NDArrayBase(object):
    """A simple Device/CPU Array object in runtime."""
    __slots__ = ["handle", "is_view"]
    # pylint: disable=no-member
    def __init__(self, handle, is_view=False):
        """Initialize the function with handle

        Parameters
        ----------
        handle : TVMArrayHandle
            the handle to the underlying C++ TVMArray
        """
        self.handle = handle
        self.is_view = is_view

    def __del__(self):
        if not self.is_view and _LIB:
            check_call(_LIB.TVMArrayFree(self.handle))

    @property
    def _tvm_handle(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value


def _make_array(handle, is_view):
    handle = ctypes.cast(handle, TVMArrayHandle)
    return _CLASS_NDARRAY(handle, is_view)

_TVM_COMPATS = ()

def _reg_extension(cls, fcreate):
    global _TVM_COMPATS
    _TVM_COMPATS += (cls,)
    if fcreate:
        fret = lambda x: fcreate(_return_handle(x))
        RETURN_SWITCH[cls._tvm_tcode] = fret
        C_TO_PY_ARG_SWITCH[cls._tvm_tcode] = _wrap_arg_func(fret, cls._tvm_tcode)


_CLASS_NDARRAY = None

def _set_class_ndarray(cls):
    global _CLASS_NDARRAY
    _CLASS_NDARRAY = cls
