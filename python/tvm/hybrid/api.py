"""APIs of lowering the Python subset to HalideIR"""
from __future__ import absolute_import as _abs

from .._ffi.base import decorate
from .. import _api_internal as _tvm_internal
from ..tensor import Tensor

from .parser import parse_python
from .util import _pruned_source


def script(pyfunc):
    """Decorate a python function function as  hybrid script.

    The hybrid function support emulation mode and parsing to
    the internal language IR.

    Returns
    -------
    hybrid_func : function
        A decorated hybrid script function.
    """
    def wrapped_func(func, *args, **kwargs): #pylint: disable=missing-docstring
        from .util import _enter_hybrid_runtime, _restore_runtime, _is_tvm_arg_types
        if _is_tvm_arg_types(args):
            src = _pruned_source(func)
            parser = parse_python(src, args)

            input_tensors = []
            for i in args:
                if isinstance(i, Tensor):
                    input_tensors.append(i)

            op = _tvm_internal._HybridOp(parser.func_name, "HybridOp", None, input_tensors,
                                         parser.outputs, parser.parsed_body)
            res = [op.output(i) for i in range(len(parser.outputs))]

            return res[0] if len(res) == 1 else res

        intersect = _enter_hybrid_runtime(func)
        value = func(*args, **kwargs)
        _restore_runtime(func, intersect)
        return value

    return decorate(pyfunc, wrapped_func)
