"""Runtime NDArray api"""
from __future__ import absolute_import

import ctypes
from ..base import _LIB, check_call
from ..runtime_ctypes import TVMArrayHandle

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
        if not self.is_view:
            check_call(_LIB.TVMArrayFree(self.handle))

    @property
    def _dltensor_addr(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

def _make_array(handle, is_view):
    handle = ctypes.cast(handle, TVMArrayHandle)
    return _CLASS_NDARRAY(handle, is_view)

_DLTENSOR_COMPATS = ()

def _reg_dltensor(cls):
    global _DLTENSOR_COMPATS
    _DLTENSOR_COMPATS += (cls,)

_CLASS_NDARRAY = None

def _set_class_ndarray(cls):
    global _CLASS_NDARRAY
    _CLASS_NDARRAY = cls
