from ._ctypes._api import _init_function_module
import _function_internal

int32 = 1
float32 = 2

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


_init_function_module("tvm.cpp")
