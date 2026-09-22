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
"""Explicit immutable bindings survive ordinary assignment unchanged."""

import pytest

import tvm
from tvm.script import parser
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as imperative
from tvm.script.ir_builder.base import BypassBind
from tvm.script.ir_builder.type_var_frame import TypeVarFrame
from tvm.tirx.script.builder import ir as native


@pytest.mark.parametrize("explicit_variable", [False, True])
def test_imperative_bind_returns_native_variable(explicit_variable):
    variable = tvm.ir.Var("explicit", "int32") if explicit_variable else None
    with IRBuilder() as builder:
        result = imperative.bind(10, T.int32, var=variable)
        assert isinstance(result, tvm.ir.Var)
        if variable is not None:
            assert result.same_as(variable)
        builder.name("bound", result)
    binding = builder.get()
    assert isinstance(binding, tvm.tirx.Bind)
    assert binding.var.same_as(result)
    assert binding.var.name == "bound"


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("explicit_name", [False, True])
def test_self_bind_preserves_native_identity_and_name(constant, explicit_name):
    with IRBuilder() as builder, TypeVarFrame() as symbols:
        signature_symbol = symbols.resolve("assigned", "int32")
        with T.function():
            x = T.arg("x", T.int32())
            expression = 10 if constant else x + 1
            variable = tvm.ir.Var("immutable", "int32") if explicit_name else None
            result = T.bind(expression, var=variable)
            assert isinstance(result, BypassBind)
            original_name = result.value.name
            # Bypass precedes annotation, rebinding, and symbol resolution policy.
            assigned = T.bind_(
                result, name="assigned", ty=object(), previous=object(), declaration=True
            )
            assert assigned is result.value
            assert assigned.name == original_name
            if variable is not None:
                assert assigned.same_as(variable)
                assert assigned.name == "immutable"
            assert not assigned.same_as(signature_symbol)
            assert symbols.resolve("assigned").same_as(signature_symbol)
            T.evaluate(assigned)
        function = builder.get()
    binding, use = function.body.seq
    assert isinstance(binding, tvm.tirx.Bind)
    assert binding.var.same_as(assigned)
    assert isinstance(use, tvm.tirx.Evaluate)
    assert use.value.same_as(assigned)


@pytest.mark.parametrize("expression", ["10", "x + (y + z)"])
@pytest.mark.parametrize("track_span", [True, False])
def test_self_bind_source_constructs_and_evaluates_once(monkeypatch, expression, track_span):
    counts = {"input": 0, "bind": 0}
    returned = []
    native_bind = native._ffi_api.Bind

    def counted_bind(*args):
        counts["bind"] += 1
        result = native_bind(*args)
        returned.append(result)
        return result

    def once(value):
        counts["input"] += 1
        return value

    monkeypatch.setattr(native._ffi_api, "Bind", counted_bind)
    source = f"""
@T.prim_func
def main(x: T.int32, y: T.int32, z: T.int32):
    assigned = T.bind(once({expression}))
    T.evaluate(assigned)
"""
    function = parser.parse(source, extra_vars={"once": once}, track_span=track_span)
    assert counts == {"input": 1, "bind": 1}
    binding, use = function.body.seq
    assert isinstance(binding, tvm.tirx.Bind)
    assert binding.var.same_as(returned[0])
    assert binding.var.name == returned[0].name
    assert isinstance(use, tvm.tirx.Evaluate)
    assert use.value.same_as(returned[0])


def test_raw_bind_still_returns_variable():
    with IRBuilder() as builder:
        with T.function():
            result = T.Bind(10, T.int32)
            assert isinstance(result, tvm.ir.Var)
            T.evaluate(result)
        function = builder.get()
    assert function.body.seq[0].var.same_as(result)
