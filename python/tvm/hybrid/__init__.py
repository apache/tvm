"""Hybrid Programming APIs of TVM Python Package.

This package maps a subset of python to HalideIR so that:
1. Users can write some preliminary versions of the computation patterns
have not been supported yet and verify it across the real execution and
python semantic emulation.
2. So far, it is a text format dedicated to HalideIR Phase 0. Refer tvm.lower
for more details. A larger ambition of this module is to support all levels of
HalideIR.
"""

# TODO(@were): Make this module more complete.
# 1. Support HalideIR dumping to Hybrid Script
# 2. Support multi-level HalideIR

from __future__ import absolute_import as _abs

from .._ffi.base import decorate
from .._ffi.function import _init_api
from ..build_module import form_body

from .module import HybridModule
from .parser import source_to_op
from .util import _pruned_source


def script(pyfunc):
    """Decorate a python function function as hybrid script.

    The hybrid function support emulation mode and parsing to
    the internal language IR.

    Returns
    -------
    hybrid_func : function
        A decorated hybrid script function.
    """
    def wrapped_func(func, *args, **kwargs): #pylint: disable=missing-docstring
        from .util import _is_tvm_arg_types
        if _is_tvm_arg_types(args):
            src = _pruned_source(func)
            return source_to_op(src, func.__globals__, args)

        from .runtime import _enter_hybrid_runtime, _restore_runtime
        intersect = _enter_hybrid_runtime(func)
        value = func(*args, **kwargs)
        _restore_runtime(func, intersect)
        return value

    return decorate(pyfunc, wrapped_func)


def build(sch, inputs, outputs, name="hybrid_func"):
    """Dump the corrent schedule to hybrid module

    Parameters
    ----------
    sch: Schedule
        The schedule to be dumped

    inputs: An array of Tensors or Vars
        The inputs of the function body

    outputs: An array of Tensors
        The outputs of the function body

    Returns
    -------
    module: HybridModule
        The built results is wrapped in a HybridModule.
        The usage of HybridModule is roughly the same as normal TVM-built modules.
    """

    stmt = form_body(sch)
    src = _Dump(stmt, inputs, outputs, name)

    return HybridModule(src, name)


_init_api("tvm.hybrid")
