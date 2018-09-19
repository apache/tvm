"""APIs of lowering the Python subset to HalideIR"""
from __future__ import absolute_import as _abs

import ast
import types
from .._ffi.base import decorate
from .._api_internal import _ExternOp
from ..api import decl_buffer
from .parser import parse_python


def script(pyfunc):
    """Decorate a python function function as  hybrid script.

    The hybrid function support emulation mode and parsing to
    the internal language IR.

    Returns
    -------
    hybrid_func : function
        A decorated hybrid script function.
    """
    def wrapped_func(func, *args, **kwargs):
        from .util import _enter_hybrid_runtime, _restore_runtime, _is_tvm_arg_types
        if _is_tvm_arg_types(args):
            return lower(func, args)


        intersect = _enter_hybrid_runtime(func)
        value = func(*args, **kwargs)
        _restore_runtime(func, intersect)
        return value
    return decorate(pyfunc, wrapped_func)


#TODO: make it False later
def lower(func, args, simple_mode=False):
    """Parse a subset of Python to HalideIR

    Parameters
    ----------
    func : str or types.FunctionType
        If it is a string, parse the source code
        If it is a function, parse the function

    args : list of Buffer or Tensor or Var
        The argument lists to the function.
        Leave it None if no buffer is related to the function to be parsed

    Returns
    -------
    root : Stmt
        The result Halide IR and the parser class instance.
    """
    from .util import _pruned_source
    if isinstance(func, str):
        src = func
    else:
        assert isinstance(func, types.FunctionType)
        src = _pruned_source(func)
    parser = parse_python(src, args)
    if simple_mode:
        return parser.parsed_body

    input_tensors = args
    input_buffers = []
    output_buffers = []
    tensor_to_buffer = {}
    for i in args:
        buf_ = tensor_to_buffer[i.name] = decl_buffer(i.shape, i.dtype, i.name)
        input_buffers.append(buf_)
    for i in parser.outputs:
        output_buffers.append(tensor_to_buffer[i])
    res = _ExternOp(parser.func_name, "HybridOp", {}, args,
            input_buffers, output_buffers, parser.parsed_body)
    return res
