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
"""Hybrid Script Parser Intrinsic Functions

IRNodes (StmtNodes without body, PrimExprNodes and more) are called intrins
"""
# pylint: disable=redefined-builtin
import tvm.tir
from .registry import register_intrin


@register_intrin
def bool(imm):
    return tvm.tir.const(imm.value, "bool")


@register_intrin
def int8(imm):
    return tvm.tir.const(imm.value, "int8")


@register_intrin
def int16(imm):
    return tvm.tir.const(imm.value, "int16")


@register_intrin
def int32(imm):
    return tvm.tir.const(imm.value, "int32")


@register_intrin
def int64(imm):
    return tvm.tir.const(imm.value, "int64")


@register_intrin
def uint8(imm):
    return tvm.tir.const(imm.value, "uint8")


@register_intrin
def uint16(imm):
    return tvm.tir.const(imm.value, "uint16")


@register_intrin
def uint32(imm):
    return tvm.tir.const(imm.value, "uint32")


@register_intrin
def uint64(imm):
    return tvm.tir.const(imm.value, "uint64")


@register_intrin
def float8(imm):
    return tvm.tir.const(imm.value, "float8")


@register_intrin
def float16(imm):
    return tvm.tir.const(imm.value, "float16")


@register_intrin
def float32(imm):
    return tvm.tir.const(imm.value, "float32")


@register_intrin
def float64(imm):
    return tvm.tir.const(imm.value, "float64")


@register_intrin
def floordiv(x, y):
    return tvm.tir.floordiv(x, y)


@register_intrin
def floormod(x, y):
    return tvm.tir.floormod(x, y)


@register_intrin
def load(dtype, var, index, predicate=True):
    return tvm.tir.Load(dtype, var, index, predicate)


@register_intrin
def cast(dtype, value):
    return tvm.tir.Cast(dtype, value)


@register_intrin
def ramp(base, stride, lanes):
    lanes = lanes.value if not isinstance(lanes, int) else lanes
    return tvm.tir.Ramp(base, stride, lanes)


@register_intrin
def broadcast(value, lanes):
    lanes = lanes.value if not isinstance(lanes, int) else lanes
    return tvm.tir.Broadcast(value, lanes)


@register_intrin
def evaluate(value):
    return tvm.tir.Evaluate(value)


@register_intrin
def store(var, index, value, predicate=True):
    return tvm.tir.Store(var, value, index, predicate)


@register_intrin
def iter_var(var, dom, iter_type, thread_tag):
    iter_type = getattr(tvm.tir.IterVar, iter_type)
    return tvm.tir.IterVar(dom, var, iter_type, thread_tag)
