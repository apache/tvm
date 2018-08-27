"""Internal utilities for parsing Python subset to HalideIR"""

import ast
import inspect
import numpy
from .intrin import HYBRID_GLOBALS
from .._ffi.base import numeric_types
from .. import api as _api
from .. import make as _make
from .. import expr as _expr
from ..tensor import Tensor


#pylint: disable=invalid-name
np_arg_types = tuple(list(numeric_types) + [numpy.ndarray])
tvm_arg_types = (Tensor, _expr.Var)
halide_imm_types = (_expr.IntImm, _expr.FloatImm, _expr.UIntImm)


# Useful constants. In avoid of runtime dependences, we use function calls to return them.
def make_nop():
    """Returns a 'no operation' node in HalideIR."""
    return _make.Evaluate(_api.const(0, dtype='int32'))


def is_docstring(node):
    """Checks if a Python AST node is a docstring"""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)


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
        raise ValueError("Expect a numpy type but % get!" % str(type(args[0])))
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
