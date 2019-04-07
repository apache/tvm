# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Intrinsics of TVM-Python Hybrid Script for Python compilation time
semantic support."""

from .. import api as _api
from .. import expr as _expr
from .. import make as _make
from .. import target as _tgt
from ..container import Array
from .. import ir_pass
from ..stmt import For
from .util import _internal_assert
from ..intrin import call_pure_intrin

#pylint: disable=redefined-builtin

LOOP_INTRIN = {
    'range'       : For.Serial,
    'unroll'      : For.Unrolled,
    'parallel'    : For.Parallel,
    'vectorize'   : For.Vectorized,
    'const_range' : (For.Unrolled, ),
}


def _range(annotation, args):
    """Handling TVM loop types"""
    n = args.__len__()
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


range = unroll = vectorize = parallel = const_range = _range #pylint: disable=invalid-name


def bind(func_id, args):
    """Handling TVM thread binding"""
    _internal_assert(func_id == "bind", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() == 2, "A loop bind should only have 2 arguments!")
    _internal_assert(isinstance(args[0], str), \
                     "A loop bind's first argument should be a string!")
    low, ext = _api.const(0, "int32"), args[1]
    iter_var = _api.thread_axis((low, ext), args[0])
    for_type = None
    return iter_var, low, ext, for_type


def _math_intrin(func_id, args):
    from .. import intrin
    return getattr(intrin, func_id)(*args)

sqrt = log = exp = tanh = sigmoid = power = popcount = _math_intrin #pylint: disable=invalid-name


def _min_max(func_id, args):
    _internal_assert(args.__len__() == 2, "Max/Min function should have 2 elements")
    return getattr(_make, func_id.title())(args[0], args[1])


min = max = _min_max #pylint: disable=invalid-name


def _allocate_tensor(func_id, args):
    """Handling TVM tensor allocation.
    You may refer hybrid.intrin.allocate for more details."""
    n = args.__len__()
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


def len(func_id, args):
    """Iterpret the len function"""
    _internal_assert(args.__len__() == 1, "Only 1 argument is expected!")
    _internal_assert(func_id == "len", "This function cannot be directly invoked!")
    try:
        return _api.convert(args[0].__len__())
    except: #pylint: disable=bare-except
        _internal_assert(args[0].shape.__len__() == 1, "Only one-dimension array can get len")
        return _api.convert(args[0].shape[0])


def _cast(func_id, args):
    _internal_assert(args.__len__() == 1 and isinstance(args[0], _expr.Expr), \
                     "Only one expression can be cast")
    return _make.Cast(func_id, args[0])

float16 = float32 = float64 = _cast #pylint: disable=invalid-name
int8 = int16 = int32 = int64 = _cast #pylint: disable=invalid-name
uint8 = uint16 = uint32 = uint64 = _cast #pylint: disable=invalid-name


def ceil_div(func_id, args):
    _internal_assert(func_id == "ceil_div", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() == 2, "2 arguments expected for division!")
    _internal_assert(isinstance(args[0], _expr.Expr), "Only expressions can div")
    _internal_assert(isinstance(args[1], _expr.Expr), "Only expressions can div")
    a, b = args[0], args[1]
    return (a + b - 1) // b


def likely(func_id, args):
    _internal_assert(args.__len__() == 1, \
                     "Only one expression can be likely")
    _internal_assert(func_id == "likely", "This function cannot be directly invoked!")
    return call_pure_intrin(args[0].dtype, 'likely', *args)


def max_num_threads(func_id, args):
    _internal_assert(func_id == "max_num_threads", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() <= 1, "At most one argument accepted!")
    if args.__len__() == 0:
        res = _tgt.current_target().max_num_threads
    else:
        _internal_assert(isinstance(args[0], _expr.UIntImm), "In tvm bool should be uint")
        res = _tgt.current_target(args[0].value).max_num_threads
    return _api.convert(res)
