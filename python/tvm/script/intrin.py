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
import tvm.tir
from .registry import register
from .utils import get_param_list


class Intrin:
    def __init__(self, intrin, stmt=False):
        self.intrin = intrin
        self.stmt = stmt

    def signature(self):
        return "tir." + self.intrin.__name__, get_param_list(self.intrin)

    def handle(self, arg_list):
        return self.intrin(*arg_list)


@register
def bool(imm):
    return tvm.tir.const(imm, "bool")


@register
def int8(imm):
    return tvm.tir.const(imm, "int8")


@register
def int16(imm):
    return tvm.tir.const(imm, "int16")


@register
def int32(imm):
    return tvm.tir.const(imm, "int32")


@register
def int64(imm):
    return tvm.tir.const(imm, "int64")


@register
def uint8(imm):
    return tvm.tir.const(imm, "uint8")


@register
def uint16(imm):
    return tvm.tir.const(imm, "uint16")


@register
def uint32(imm):
    return tvm.tir.const(imm, "uint32")


@register
def uint64(imm):
    return tvm.tir.const(imm, "uint64")


@register
def float8(imm):
    return tvm.tir.const(imm, "float8")


@register
def float16(imm):
    return tvm.tir.const(imm, "float16")


@register
def float32(imm):
    return tvm.tir.const(imm, "float32")


@register
def float64(imm):
    return tvm.tir.const(imm, "float64")


@register
def floordiv(x, y):
    return tvm.tir.floordiv(x, y)


@register
def floormod(x, y):
    return tvm.tir.floormod(x, y)


@register
def load(dtype, var, index, predicate=True):
    return tvm.tir.Load(dtype, var, index, predicate)


@register
def cast(value, dtype):
    return tvm.tir.Cast(dtype, value)


@register
def ramp(base, stride, lanes):
    return tvm.tir.Ramp(base, stride, lanes)


@register
def broadcast(value, lanes):
    return tvm.tir.Broadcast(value, lanes)


@register
def iter_var(var, dom, iter_type, thread_tag):
    iter_type = getattr(tvm.tir.IterVar, iter_type)
    return tvm.tir.IterVar(dom, var, iter_type, thread_tag)


@register
def max(a, b):  # pylint: disable=redefined-builtin
    return tvm.tir.Max(a, b)


def get_axis(begin, end, iter_type):
    ana = tvm.arith.Analyzer()
    extent = ana.simplify(end - begin)
    block_var_dom = tvm.ir.Range.from_min_extent(begin, extent)

    iter_type_dict = {"data_par": 0, "reduce": 2, "scan": 3, "opaque": 4}
    return tvm.tir.IterVar(block_var_dom, "bv", iter_type_dict[iter_type])


@register
def range(begin, end):
    return get_axis(begin, end, "data_par")


@register
def reduce_axis(begin, end):
    return get_axis(begin, end, "reduce")


@register
def scan_axis(begin, end):
    return get_axis(begin, end, "scan")


@register
def opaque_axis(begin, end):
    return get_axis(begin, end, "opaque")


@register
class EvaluateIntrin(Intrin):
    def __init__(self):
        def evaluate(value):
            return tvm.tir.Evaluate(value)

        super().__init__(evaluate, stmt=True)


@register
class StoreIntrin(Intrin):
    def __init__(self):
        def store(var, index, value, predicate=True):
            return tvm.tir.Store(var, value, index, predicate)

        super().__init__(store, stmt=True)
