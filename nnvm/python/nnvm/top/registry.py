# pylint: disable=invalid-name
"""Information registry to register operator information for compiler"""
import tvm

class OpPattern(object):
    """Operator generic patterns

    See Also
    --------
    top.tag : Contains explanation of the tag type.
    """
    # Elementwise operator
    ELEMWISE = 0
    # Broadcast operator
    BROADCAST = 1
    # Injective mapping
    INJECTIVE = 2
    # Comunication
    COMM_REDUCE = 3
    # Complex op, can still fuse ewise into it
    OUT_ELEMWISE_FUSABLE = 4
    # Not fusable opaque op
    OPAQUE = 8

_register_compute = tvm.get_global_func("nnvm._register_compute")
_register_schedule = tvm.get_global_func("nnvm._register_schedule")
_register_pattern = tvm.get_global_func("nnvm._register_pattern")
_register_alter_op_layout = tvm.get_global_func("nnvm.compiler._register_alter_op_layout")

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


def register_alter_op_layout(op_name, f=None, level=10):
    """Register alter layout function for operator

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
        _register_alter_op_layout(op_name, myf, level)
        return myf
    return register(f) if f else register
