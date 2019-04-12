"""Custom datatype functionality"""
from __future__ import absolute_import as _abs

from ._ffi.function import register_func as _register_func
from . import make as _make
from .api import convert
from .expr import Call as _Call

def _register_op(op_name, target, datatype_name, extern_func_name):
    """Register an external function which computes the given op.

    Parameters
    ----------
    target : str
        The name of codegen target.

    op_name : str
        The name of the operation which the function computes, given by its
        Halide::Internal class name (e.g. Add, LE, Not).

    datatype_name : str
        The name of the custom datatype.

    extern_func_name : str
        The name of the external function to call.
    """

    def create_op_to_call_lower(extern_func_name):
        """Create a function which lowers an operation to a function call."""
        def lower(op):
            print(op.dtype)
            # TODO(gus) need to make bits available by parsing the type string
            # return _make.Call(
            #     "uint" + str(op.dtype.bits), extern_func_name,
            #     convert([op.a, op.b]), _Call.Extern, None, 0)
        return lower

    lower_func_name = "tvm.datatype.lower." + target + "." + op_name + "." \
                      + datatype_name
    _register_func(lower_func_name, create_op_to_call_lower(extern_func_name))
