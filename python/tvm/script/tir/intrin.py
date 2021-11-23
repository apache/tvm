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
"""TVM Script Parser Intrinsic Classes"""
# pylint: disable=redefined-builtin, relative-beyond-top-level
import builtins
from typing import List, Optional, Any

import tvm.tir
from tvm.ir import Span
from tvm.tir.sparse import (
    Axis,
    DenseFixedAxis,
    DenseVariableAxis,
    SpIterVar,
    SparseFixedAxis,
    SparseVariableAxis,
)
from ..registry import register
from ..utils import get_param_list, tvm_span_from_synr


class Intrin:
    def __init__(self, intrin, stmt=False):
        self.intrin = intrin
        self.stmt = stmt

    def signature(self):
        return "tir." + self.intrin.__name__, get_param_list(self.intrin)

    def handle(self, arg_list: List[Any], span: tvm.ir.Span):
        return self.intrin(*arg_list, span=tvm_span_from_synr(span))


@register
def bool(imm, span):
    return imm.astype("bool", span)


@register
def int8(imm, span):
    return imm.astype("int8", span)


@register
def int16(imm, span):
    return imm.astype("int16", span)


@register
def int32(imm, span):
    return imm.astype("int32", span)


@register
def int64(imm, span):
    return imm.astype("int64", span)


@register
def uint8(imm, span):
    return imm.astype("uint8", span)


@register
def uint16(imm, span):
    return imm.astype("uint16", span)


@register
def uint32(imm, span):
    return imm.astype("uint32", span)


@register
def uint64(imm, span):
    return imm.astype("uint64", span)


@register
def float8(imm, span):
    return imm.astype("float8", span)


@register
def float16(imm, span):
    return imm.astype("float16", span)


@register
def float32(imm, span):
    return imm.astype("float32", span)


@register
def float64(imm, span):
    return imm.astype("float64", span)


@register
def min_value(dtype, span):
    return tvm.tir.min_value(dtype, span)


@register
def max_value(dtype, span):
    return tvm.tir.max_value(dtype, span)


@register
def lower_bound(arr, val, l, r, span):
    return tvm.tir.lower_bound(arr, val, l, r, span)


@register
def upper_bound(arr, val, l, r, span):
    return tvm.tir.upper_bound(arr, val, l, r, span)


@register
def atomic_add(ptr, val, span):
    return tvm.tir.atomic_add(ptr, val, span)


@register
def floordiv(x, y, span):
    return tvm.tir.floordiv(x, y, span)


@register
def floormod(x, y, span):
    return tvm.tir.floormod(x, y, span)


@register
def truncmod(x, y, span):
    return tvm.tir.truncmod(x, y, span)


@register
def abs(x, span):
    return tvm.tir.abs(x, span)


@register
def load(dtype, var, index, predicate=None, span=None):
    return tvm.tir.Load(dtype, var, index, predicate, span)


@register
def cast(value, dtype, span):
    return tvm.tir.Cast(dtype, value, span)


@register
def ramp(base, stride, lanes, span):
    return tvm.tir.Ramp(base, stride, lanes.value, span)


@register
def broadcast(value, lanes, span):
    return tvm.tir.Broadcast(value, lanes.value, span)


@register
def iter_var(var, dom, iter_type, thread_tag, span):
    iter_type = getattr(tvm.tir.IterVar, iter_type)
    return tvm.tir.IterVar(dom, var, iter_type, thread_tag, span)


@register
def max(a, b, span):  # pylint: disable=redefined-builtin
    return tvm.tir.Max(a, b, span)


@register
def min(a, b, span):  # pylint: disable=redefined-builtin
    return tvm.tir.Min(a, b, span)


def get_axis(begin, end, iter_type, span):
    ana = tvm.arith.Analyzer()
    extent = ana.simplify(end - begin)
    block_var_dom = tvm.ir.Range.from_min_extent(begin, extent)

    iter_type_dict = {"data_par": 0, "reduce": 2, "scan": 3, "opaque": 4}
    return tvm.tir.IterVar(block_var_dom, "bv", iter_type_dict[iter_type], span=span)


@register
def range(begin, end, span):
    return get_axis(begin, end, "data_par", span)


@register
def reduce_axis(begin, end, span):
    return get_axis(begin, end, "reduce", span)


@register
def scan_axis(begin, end, span):
    return get_axis(begin, end, "scan", span)


@register
def opaque_axis(begin, end, span):
    return get_axis(begin, end, "opaque", span)


@register
def Select(cond, if_body, else_body, span):  # pylint: disable=invalid-name
    return tvm.tir.Select(cond, if_body, else_body, span)


@register
class EvaluateIntrin(Intrin):
    def __init__(self):
        def evaluate(value, span):
            return tvm.tir.Evaluate(value, span)

        super().__init__(evaluate, stmt=True)


@register
class StoreIntrin(Intrin):
    def __init__(self):
        def store(var, index, value, predicate=True, span=None):
            return tvm.tir.Store(var, value, index, predicate, span)

        super().__init__(store, stmt=True)


@register
def comm_reducer(lambda_io, identities, span):
    """Create a CommReducer from lambda inputs/outputs and the identities"""
    lambda_input = lambda_io[0]
    lambda_output = lambda_io[1]

    num_args = len(lambda_input)
    num_arg_per_group = num_args // 2
    x = [lambda_input[i] for i in builtins.range(0, num_arg_per_group)]
    y = [lambda_input[i] for i in builtins.range(num_arg_per_group, num_args)]

    if not isinstance(lambda_output, tuple):
        lambda_output = (lambda_output,)

    return tvm.tir.CommReducer(x, y, lambda_output, identities, span)


@register
def to_dense(axis: Axis, span: Optional[Span] = None):
    if isinstance(axis, (SparseFixedAxis, SparseVariableAxis)):
        return DenseFixedAxis(axis.name + "_dense", axis.length, axis)
    else:
        return axis


@register
def cord(axis: Axis, span: Optional[Span] = None):
    # The field `var` and `is_reduction` will be updated in SparseBlock scope handler
    var_temp = tvm.te.var()
    if isinstance(axis, DenseVariableAxis):
        return SpIterVar(var_temp, axis.length, SpIterVar.DenseVariable, False, axis)
    else:
        return SpIterVar(var_temp, axis.length, SpIterVar.DenseFixed, False, axis)


@register
def pos(axis: Axis, span: Optional[Span] = None):
    # The field `var` and `is_reduction` will be updated in SparseBlock scope handler
    var_temp = tvm.te.var()
    if isinstance(axis, DenseFixedAxis):
        return SpIterVar(var_temp, axis.length, SpIterVar.DenseFixed, False, axis)
    elif isinstance(axis, DenseVariableAxis):
        return SpIterVar(var_temp, axis.length, SpIterVar.DenseVariable, False, axis)
    elif isinstance(axis, SparseFixedAxis):
        return SpIterVar(var_temp, axis.nnz_cols, SpIterVar.SparseFixed, False, axis)
    else:
        return SpIterVar(var_temp, axis.length, SpIterVar.SparseVariable, False, axis)
