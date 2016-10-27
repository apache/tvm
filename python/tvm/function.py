from __future__ import absolute_import as _abs
from numbers import Number as _Number, Integral as _Integral
from ._ctypes._api import _init_function_module
from .import _function_internal
from .import make as _make

int32 = "int32"
float32 = "float32"

def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, _Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _function_internal._const(value, dtype)


def Var(name="tindex", dtype=int32):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type
    """
    return _function_internal._Var(name, dtype)


def convert(value):
    """Convert a value to expression."""
    if isinstance(value, _Number):
        return const(value)
    elif isinstance(value, list):
        value = [convert(x) for x in value]
        return _function_internal._Array(*value)
    else:
        return value

_init_function_module("tvm")
