"""Utilities for parsing Python subset to HalideIR"""

import sys
import inspect
import numpy
from ._intrin import HYBRID_GLOBALS
from .. import api as _api
from .. import make as _make
from .. import expr as _expr
from ..tensor import Tensor

# Useful constants
NOP = _make.Evaluate(_api.const(0, dtype='int32'))
RANGE_ONE = _make.range_by_min_extent(0, 1)
TRUE = _api.convert(True)
ZERO = _api.const(0)

# Node types represent constants in HalideIR
HALIDE_IMM = (expr.FloatImm, _expr.IntImm, _expr.UIntImm)

def _pruned_source(func):
    """Prune source code's extra leading spaces"""
    lines = inspect.getsource(func).split('\n')
    leading_space = len(lines[0]) - len(lines[0].lstrip(' '))
    lines = [line[leading_space:] for line in lines]
    return '\n'.join(lines)

TVM_ARG_TYPES = (_expr.Var, Tensor)
if sys.version_info[0] == 3:
    NUMPY_ARG_TYPES = (float, int, numpy.float32, numpy.int32, numpy.ndarray)
else:
    NUMPY_ARG_TYPES = (float, int, long, numpy.float32, numpy.int32, numpy.ndarray)

def _is_tvm_arg_types(args):
    """Determine a list of element is either a list of tvm arguments of a list of numpy arguments.
    If neither is true, raise a assertion error.
    """
    if isinstance(args[0], TVM_ARG_TYPES):
        for elem in args[1:]:
            assert isinstance(elem, TVM_ARG_TYPES)
        return True
    assert isinstance(args[0], NUMPY_ARG_TYPES)
    for elem in args[1:]:
        assert isinstance(elem, NUMPY_ARG_TYPES)
    return False

def _enter_hybrid_runtime(func):
    """Put hybrid runtime variables into the global scope"""
    #_globals = globals(func)
    _globals = func.__globals__
    intersect = []
    for elem in list(HYBRID_GLOBALS.keys()):
        if elem in _globals.keys():
            intersect.append((elem, _globals[elem]))
        _globals[elem] = HYBRID_GLOBALS[elem]
    return intersect

def _restore_runtime(func, intersect):
    """Rollback the modification caused by hybrid runtime"""
    #_globals = globals(func)
    _globals = func.__globals__
    for elem in list(HYBRID_GLOBALS.keys()):
        _globals.pop(elem)
    for k, v in intersect:
        _globals[k] = v
