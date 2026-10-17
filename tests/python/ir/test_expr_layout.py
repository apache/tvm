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

"""Shared expression modules and call effect values."""

import importlib

import tvm_ffi

from tvm_tirx import ir
from tvm_tirx.ir import CallEffectKind, FloatImm, IntImm
from tvm_tirx.ir.prim.vector_expr import Broadcast, Ramp, Shuffle
from tvm_tirx.ir.unique_name_supply import UniqueNameSupply


def test_scalar_expression_constructors():
    integer = IntImm(ir.PrimType("int32"), 3)
    floating = FloatImm(ir.PrimType("float32"), 2.5)
    assert int(integer) == 3
    assert bool(integer)
    assert not bool(IntImm("int32", 0))
    assert bool(integer == 3)
    assert bool(integer != 4)
    assert float(floating) == 2.5
    for value in (integer, floating):
        assert type(ir.load_json(ir.save_json(value))) is type(value)


def test_vector_expression_constructors():
    ramp = Ramp(0, 1, 4)
    broadcast = Broadcast(3, 4)
    shuffled = Shuffle([ramp, broadcast], [0, 5, 2, 7])
    for value in (ramp, broadcast, shuffled):
        restored = ir.load_json(ir.save_json(value))
        assert type(restored) is type(value)
        assert tvm_ffi.structural_equal(restored, value)


def test_expression_module_locations():
    assert ir.IntImm is IntImm
    assert ir.FloatImm is FloatImm
    assert ir.CallEffectKind is CallEffectKind
    assert IntImm.__module__ == FloatImm.__module__ == "tvm_tirx.ir.expr"
    for cls in (Ramp, Broadcast, Shuffle):
        assert cls.__module__ == "tvm_tirx.ir.prim.vector_expr"
    assert UniqueNameSupply.__module__ == "tvm_tirx.ir.unique_name_supply"
    for module_name in ("tvm_tirx.ir.prim.expr", "tvm_tirx.tirx.expr", "tvm_tirx.tirx"):
        module = importlib.import_module(module_name)
        for name in ("IntImm", "FloatImm", "Ramp", "Broadcast", "Shuffle", "CallEffectKind"):
            assert not hasattr(module, name)


def test_call_effect_kind_values_and_ffi_conversion():
    assert {kind.name: kind.value for kind in CallEffectKind} == {
        "ExprAnnotation": 0,
        "Pure": 1,
        "ReadState": 2,
        "UpdateState": 3,
        "SpecialCallArg": 4,
        "EmbedInfo": 5,
        "ControlJump": 6,
    }
    assert CallEffectKind.Opaque is CallEffectKind.UpdateState
    for kind in CallEffectKind:
        attrs = ir.DictAttrs(dict={"effect": kind})
        assert int(attrs["effect"]) == kind.value
