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

from tvm.runtime import const, convert
import tvm.te
from tvm.ir.container import Array
from tvm.target import Target
from tvm.tir import expr as _expr
from tvm.tir import call_intrin
from tvm.tir.stmt import ForKind

from .utils import _internal_assert

# pylint: disable=redefined-builtin,invalid-name

LOOP_INTRIN = {
    "range": ForKind.SERIAL,
    "unroll": ForKind.UNROLLED,
    "parallel": ForKind.PARALLEL,
    "vectorize": ForKind.VECTORIZED,
    "const_range": (ForKind.UNROLLED,),
}


def _range(annotation, args):
    """Handling TVM loop types"""
    n = args.__len__()
    if n == 1:
        low, ext = const(0, dtype="int32"), args[0]
    else:
        _internal_assert(n == 2, "A loop intrinsic should only have 1 or 2 arguments!")
        low, ext = args[0], args[1]
    if not tvm.tir.analysis.expr_deep_equal(low, const(0, dtype="int32")):
        ext = ext - low
    kind = LOOP_INTRIN[annotation]
    iter_var = None
    return iter_var, low, ext, kind


range = unroll = vectorize = parallel = const_range = _range  # pylint: disable=invalid-name


def bind(func_id, args):
    """Handling TVM thread binding"""
    _internal_assert(func_id == "bind", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() == 2, "A loop bind should only have 2 arguments!")
    _internal_assert(isinstance(args[0], str), "A loop bind's first argument should be a string!")
    low, ext = const(0, "int32"), args[1]
    iter_var = tvm.te.thread_axis((low, ext), args[0])
    kind = None
    return iter_var, low, ext, kind


def _math_intrin(func_id, args):
    # pylint: disable=import-outside-toplevel
    from tvm.tir import op

    return getattr(op, func_id)(*args)


sqrt = (
    log
) = exp = tanh = sigmoid = power = popcount = round = _math_intrin  # pylint: disable=invalid-name


def _min_max(func_id, args):
    _internal_assert(args.__len__() == 2, "Max/Min function should have 2 elements")
    return getattr(_expr, func_id.title())(args[0], args[1])


min = max = _min_max  # pylint: disable=invalid-name


def _allocate_tensor(func_id, args):
    """Handling TVM tensor allocation.
    You may refer hybrid.intrin.allocate for more details."""
    n = args.__len__()
    _internal_assert(
        isinstance(convert(args[0]), Array), "allocate's first argument should be a tuple of shape!"
    )
    shape = args[0]
    for i in shape:
        _internal_assert(isinstance(i, (_expr.PrimExpr, int)), "The shape should be an expression")
    if n > 1:
        _internal_assert(isinstance(args[1], str), "The data type should be an str")
        _internal_assert(
            args[1].startswith("int") or args[1].startswith("float"),
            "The data type should be either int or float!",
        )
        dtype = args[1]
    else:
        dtype = "float32"
    if n > 2:
        _internal_assert(isinstance(args[2], str), "The data scope should be an string")
        _internal_assert(func_id != "output_tensor", "Output tensor cannot specify scope")
        scope = args[2]
    else:
        scope = "global" if func_id != "output_tensor" else "output"
    return (shape, dtype, scope)


output_tensor = allocate = _allocate_tensor  # pylint: disable=invalid-name


def len(func_id, args):
    """Iterpret the len function"""
    _internal_assert(args.__len__() == 1, "Only 1 argument is expected!")
    _internal_assert(func_id == "len", "This function cannot be directly invoked!")
    try:
        return convert(args[0].__len__())
    except:  # pylint: disable=bare-except
        _internal_assert(args[0].shape.__len__() == 1, "Only one-dimension array can get len")
        return convert(args[0].shape[0])


def _cast(func_id, args):
    _internal_assert(
        args.__len__() == 1,
        f"Casting to {func_id} only supports a single argument",
    )
    # The FFI can handle any conversion of `args[0]` into PrimExpr, if
    # required.
    return _expr.Cast(func_id, args[0])


float16 = float32 = float64 = _cast  # pylint: disable=invalid-name
int8 = int16 = int32 = int64 = _cast  # pylint: disable=invalid-name
uint8 = uint16 = uint32 = uint64 = _cast  # pylint: disable=invalid-name


def ceil_div(func_id, args):
    _internal_assert(func_id == "ceil_div", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() == 2, "2 arguments expected for division!")
    a, b = args
    return (a + b - 1) // b


def likely(func_id, args):
    _internal_assert(args.__len__() == 1, "Only one expression can be likely")
    _internal_assert(func_id == "likely", "This function cannot be directly invoked!")
    return call_intrin(args[0].dtype, "tir.likely", *args)


def max_num_threads(func_id, args):
    """Set the maximum number of threads."""
    _internal_assert(func_id == "max_num_threads", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() <= 1, "At most one argument accepted!")
    if args.__len__() == 0:
        res = Target.current().max_num_threads
    else:
        _internal_assert(isinstance(args[0], _expr.IntImm), "In tvm bool should be uint")
        res = Target.current(args[0].value).max_num_threads
    return convert(res)


def inf(func_id, args):
    """Infinity"""
    _internal_assert(func_id == "inf", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() == 1, "One argument accepted!")
    return tvm.tir.max_value(args[0])


def ninf(func_id, args):
    """Negative infinity"""
    _internal_assert(func_id == "ninf", "This function cannot be directly invoked!")
    _internal_assert(args.__len__() == 1, "One argument accepted!")
    return tvm.tir.min_value(args[0])
