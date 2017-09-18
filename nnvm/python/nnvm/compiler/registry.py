# pylint: disable=invalid-name
"""Information registry to register operator information for compiler"""
import tvm

class OpPattern(object):
    ELEM_WISE = 0
    BROADCAST = 1
    COMPLEX = 2
    EXTERN = 2

_register_compute = tvm.get_global_func("nnvm._register_compute")
_register_schedule = tvm.get_global_func("nnvm._register_schedule")
_register_pattern = tvm.get_global_func("nnvm._register_pattern")

def register_compute(op_name, f=None, level=10):
    """Register compute function for operator

    Parameters
    ----------
    op_name : str
        The name of operator

    f : function
        The schedule function

    level : int
        The priority level

    Returns
    -------
    fregister : function
        Register function if f is not specified.
    """
    def register(myf):
        """internal register function"""
        _register_compute(op_name, myf, level)
        return myf
    return register(f) if f else register


def register_schedule(op_name, f=None, level=10):
    """Register schedule function for operator

    Parameters
    ----------
    op_name : str
        The name of operator

    f : function
        The schedule function

    level : int
        The priority level

    Returns
    -------
    fregister : function
        Register function if f is not specified.
    """
    def register(myf):
        """internal register function"""
        _register_schedule(op_name, myf, level)
        return myf
    return register(f) if f else register


def register_pattern(op_name, pattern, level=10):
    """Register pattern code for operator

    Parameters
    ----------
    op_name : str
        The name of operator

    pattern : int
        The pattern code.

    level : int
        The priority level
    """
    _register_pattern(op_name, pattern, level)
