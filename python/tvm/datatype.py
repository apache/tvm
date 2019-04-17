"""Custom datatype functionality"""
from __future__ import absolute_import as _abs

from ._ffi.function import register_func as _register_func
from . import make as _make
from .api import convert
from .expr import Call as _Call, Cast as _Cast
from ._ffi.runtime_ctypes import TVMType as _TVMType
from . import _api_internal


def _datatype_register_op(extern_func_name,
                          op_name,
                          target,
                          datatype_name,
                          src_datatype_name=None):
    """Register an external function which computes the given op.

    Currently, this will only work with Casts and binary expressions
    whose arguments are named `a` and `b`.
    TODO(gus) figure out what other special cases must be handled by
        looking through expr.py.

    Parameters
    ----------
    extern_func_name : str
        The name of the external function to call.

    op_name : str
        The name of the operation which the function computes, given by its
        Halide::Internal class name (e.g. Add, LE, Not).

    target : str
        The name of codegen target.

    datatype_name : str
        The name of the custom datatype, e.g. posit (but not custom[posit]8).

    src_datatype_name : str
        If op_name is "Cast", then this should be set to the source datatype of
        the argument to the Cast. If op_name is not "Cast", this is unused.
    """

    def lower(op):
        """Function which lowers an operation to a function call.

        Takes an op---either a Cast or a binary op (e.g. an Add) and returns a
        call to the specified external function, passing the op's argument
        (Cast) or arguments (a binary op). The return type of the call depends
        on the type of the op: if it is a custom type, then a uint of the same
        width as the custom type is returned. Otherwise, the type is
        unchanged."""
        dtype = op.dtype
        t = _TVMType(dtype)
        if _api_internal._datatype_registered(t.type_code):
            dtype = "uint" + str(t.bits)
            if t.lanes > 1:
                dtype += "x" + str(t.lanes)
        if type(op) is _Cast:
            return _make.Call(dtype, extern_func_name, convert([op.value]),
                              _Call.Extern, None, 0)
        else:
            return _make.Call(dtype, extern_func_name, convert([op.a, op.b]),
                              _Call.Extern, None, 0)

    if op_name is "Cast":
        assert (src_datatype_name is not None)
        lower_func_name = "tvm.datatype.lower." + target + "." + op_name + "." \
                          + datatype_name + "." + src_datatype_name
    else:
        lower_func_name = "tvm.datatype.lower." + target + "." + op_name + "." \
                          + datatype_name
    _register_func(lower_func_name, lower)
