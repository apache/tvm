"""Internal utilities for parsing Python subset to HalideIR"""

import sys
import inspect
import numpy
from ._intrin import HYBRID_GLOBALS
from .._ffi.base import np_arg_types
from .. import api as _api
from .. import make as _make
from .. import expr as _expr
from ..tensor import Tensor

# If it is a 
tvm_arg_types = (Tensor, _expr.Var)
halide_imm_types = (_expr.IntImm, _expr.FloatImm, _expr.UIntImm)

# Useful constants
def make_nop():
    return _make.Evaluate(_api.const(0, dtype='int32'))

def make_range_one():
    return _make.range_by_min_extent(0, 1)

def make_const_true():
    return _api.convert(True)

def _pruned_source(func):
    """Prune source code's extra leading spaces"""
    lines = inspect.getsource(func).split('\n')
    leading_space = len(lines[0]) - len(lines[0].lstrip(' '))
    lines = [line[leading_space:] for line in lines]
    return '\n'.join(lines)

def _is_tvm_arg_types(args):
    """Determine a list of element is either a list of tvm arguments of a list of numpy arguments.
    If neither is true, raise a value error."""
    if isinstance(args[0], tvm_arg_types):
        for elem in args[1:]:
            if not isinstance(elem, tvm_arg_types):
                raise ValueError("Expect a Var or Tensor instance but % get!" % str(type(elem)))
        return True
    if not isinstance(args[0], np_arg_types):
        raise ValueError("Expect a numpy type but % get!" % str(type(elem)))
    for elem in args[1:]:
        if not isinstance(elem, np_arg_types):
            raise ValueError("Expect a numpy type but % get!" % str(type(elem)))
    return False

def _enter_hybrid_runtime(func):
    """Put hybrid runtime variables into the global scope"""
    _globals = func.__globals__
    intersect = []
    for elem in list(HYBRID_GLOBALS.keys()):
        if elem in _globals.keys():
            intersect.append((elem, _globals[elem]))
        _globals[elem] = HYBRID_GLOBALS[elem]
    return intersect

def _restore_runtime(func, intersect):
    """Rollback the modification caused by hybrid runtime"""
    _globals = func.__globals__
    for elem in list(HYBRID_GLOBALS.keys()):
        _globals.pop(elem)
    for k, v in intersect:
        _globals[k] = v
