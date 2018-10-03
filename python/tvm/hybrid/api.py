"""APIs of lowering the Python subset to HalideIR"""
from __future__ import absolute_import as _abs

import ast
import types
from .._ffi.base import decorate
from .._api_internal import _ExternOp
from ..api import decl_buffer
from ..expr import Call
from ..stmt import Provide
from ..ir_pass import PostOrderVisit
from ..tensor import Tensor
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
    body = parser.parsed_body
    if simple_mode:
        return body

    input_tensors = set()
    output_tensors = set()
    def find_inputs(op):
        if isinstance(op, Provide):
            for i in args:
                if op.func.name == i.name:
                    output_tensors.add(i.name)
        elif isinstance(op, Call):
            for i in args:
                if op.name == i.name and (i.name not in output_tensors):
                    input_tensors.add(i)
    PostOrderVisit(body, find_inputs)
    input_tensors = list(input_tensors)

    input_buffers = []
    output_buffers = []
    attrs = {}
    for i in args:
        if isinstance(i, Tensor):
            buf_ = decl_buffer(i.shape, i.dtype, i.name)
            if i in input_tensors:
                input_buffers.append(buf_)
            if i.name in output_tensors:
                output_buffers.append(buf_)
        attrs[i.name] = i

    res = _ExternOp(parser.func_name, "HybridOp", attrs, input_tensors,
            input_buffers, output_buffers, body)
    return res
