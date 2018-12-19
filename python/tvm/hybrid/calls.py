"""Intrinsics of TVM-Python Hybrid Script for Python compilation time
semantic support."""

from .. import api as _api
from .. import expr as _expr
from .. import make as _make
from ..container import Array
from .. import ir_pass
from ..stmt import For
from .util import _internal_assert

#pylint: disable=redefined-builtin

LOOP_INTRIN = {
    'range'    : For.Serial,
    'unroll'   : For.Unrolled,
    'parallel' : For.Parallel,
    'vectorize': For.Vectorized,
}

def _range(annotation, args):
    """Handling TVM loop types"""
    n = len(args)
    if n == 1:
        low, ext = _api.const(0, dtype='int32'), args[0]
    else:
        _internal_assert(n == 2, "A loop intrinsic should only have 1 or 2 arguments!")
        low, ext = args[0], args[1]
    if not ir_pass.Equal(low, _api.const(0, dtype='int32')):
        ext = ext - low
    for_type = LOOP_INTRIN[annotation]
    iter_var = None
    return iter_var, low, ext, for_type


range = unroll = vectorize = parallel = _range #pylint: disable=invalid-name


def bind(func_id, args):
    """Handling TVM thread binding"""
    _internal_assert(func_id == "bind", "This function cannot be directly invoked!")
    _internal_assert(len(args) == 2, "A loop bind should only have 2 arguments!")
    _internal_assert(isinstance(args[0], str), \
                     "A loop bind's first argument should be a string!")
    iter_var = _api.thread_axis(args[0])
    low, ext = _api.const(0), args[1]
    for_type = None
    return iter_var, low, ext, for_type


def _math_intrin(func_id, args):
    from .. import intrin
    return getattr(intrin, func_id)(*args)

sqrt = log = exp = tanh = sigmoid = power = popcount = _math_intrin #pylint: disable=invalid-name


def _min_max(func_id, args):
    _internal_assert(len(args) == 2, "Max/Min function should have 2 elements")
    return getattr(_make, func_id.title())(args[0], args[1])


min = max = _min_max #pylint: disable=invalid-name


def _allocate_tensor(func_id, args):
    """Handling TVM tensor allocation.
    You may refer hybrid.intrin.allocate for more details."""
    n = len(args)
    _internal_assert(isinstance(_api.convert(args[0]), Array), \
                     "allocate's first argument should be a tuple of shape!")
    shape = args[0]
    for i in shape:
        _internal_assert(isinstance(i, _expr.Expr), "The shape should be an expression")
    if n > 1:
        _internal_assert(isinstance(args[1], str),
                         "The data type should be an str")
        _internal_assert(args[1].startswith('int') or args[1].startswith('float'), \
                         "The data type should be either int or float!")
        dtype = args[1]
    else:
        dtype = 'float32'
    if n > 2:
        _internal_assert(isinstance(args[2], str), \
                         "The data scope should be an string")
        _internal_assert(func_id != 'output_tensor', "Output tensor cannot specify scope")
        scope = args[2]
    else:
        scope = 'global' if func_id != 'output_tensor' else 'output'
    return (shape, dtype, scope)

output_tensor = allocate = _allocate_tensor #pylint: disable=invalid-name
