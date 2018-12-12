"""Intrinsics of TVM-Python Hybrid Script for Python compilation time"""

import ast
from .. import api as _api
from .. import expr as _expr
from .. import make as _make
from ..ir_pass import Equal
from ..stmt import For
from .util import _internal_assert

LOOP_INTRIN = {
    'range'    : For.Serial,
    'unroll'   : For.Unrolled,
    'parallel' : For.Parallel,
    'vectorize': For.Vectorized,
}

def _range(visitor, func_id, args):
    """Handling TVM loop types"""
    n = len(args)
    if n == 1:
        low, ext = _api.const(0, dtype='int32'), visitor.visit(args[0])
    else:
        _internal_assert(n == 2, "A loop intrinsic should only have 1 or 2 arguments!")
        low, ext = visitor.visit(args[0]), visitor.visit(args[1])
    if not Equal(low, _api.const(0, dtype='int32')):
        ext = ext - low
    for_type = LOOP_INTRIN[func_id]
    iter_var = None
    return iter_var, low, ext, for_type


range = unroll = vectorize = parallel = _range #pylint: disable=invalid-name


def bind(visitor, func_id, args):
    """Handling TVM thread binding"""
    n = len(args)
    _internal_assert(n == 2, "A loop bind should only have 2 arguments!")
    _internal_assert(isinstance(args[0], ast.Str), \
                     "A loop bind's first argument should be a string!")
    _vn = args[0].s
    iter_var = thread_axis(args[0].s)
    low, ext = _api.const(0, dtype='int32'), visitor.visit(args[1])
    for_type = None
    return iter_var, low, ext, for_type


def _math_intrin(visitor, func_id, args):
    from .. import intrin
    return getattr(intrin, func_id)(*[visitor.visit(arg) for arg in args])

sqrt = log = exp = tanh = sigmoid = power = popcount = _math_intrin

def _min_max(self, func_id, args):
    n = len(args)
    _internal_assert(n == 2, "Max/Min function should have 2 elements")
    a, b = self.visit(args[0]), self.visit(args[1])
    return getattr(_make, func_id.title())(a, b)

min = max = _min_max

def _allocate_tensor(visitor, func_id, args):
    """Handling TVM tensor allocation"""
    n = len(args)
    _internal_assert(isinstance(args[0], ast.Tuple), \
                     "allocate's first argument should be a tuple of shape!")
    shape = tuple(visitor.visit(i) for i in args[0].elts)
    if func_id == 'output_tensor':
        _internal_assert(not visitor.loops_above, \
                         "Are you sure to allocate a output buffer multiple times?")
    for i in shape:
        _internal_assert(isinstance(i, _expr.Expr), "The shape should be an expression")
    if n > 1:
        if isinstance(args[1], ast.Str):
            dtype = args[1].s
        else:
            _internal_assert(isinstance(args[1], ast.Attribute), \
                             "Unable to evaluate to get data type")
            to_eval = args[1]
            _internal_assert(isinstance(to_eval.value, ast.Name), \
                             "Unable to evaluate the attribute to get data type")
            _internal_assert(to_eval.attr == 'dtype', \
                             "Only dtype attribute is supported so far")
            dtype = visitor._get_buffer_from_id(to_eval.value.id).dtype
    else:
        dtype = 'float32'
    if n > 2:
        _internal_assert(isinstance(args[2], ast.Str), \
                         "The data scope should be an string")
        _internal_assert(func_id != 'output_tensor', "Output tensor cannot specify scope")
        scope = args[2].s
    else:
        scope = 'global' if func_id != 'output_tensor' else 'output'
    return (shape, dtype, scope)

output_tensor = allocate = _allocate_tensor
