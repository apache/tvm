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
"""Explicit source emission and generated statement consumption agree."""

import pytest

import tvm
from tvm import ir, tirx
from tvm.relax.script import builder as R
from tvm.relax.script.builder import ir as native_R
from tvm.script import parser
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import parser_support as PS
from tvm.tirx.script import builder as tir_builder


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize(
    "body",
    [
        "v: T.int32 = 1",
        "v: T.int32\n    v = v + 1",
        'A = T.alloc_buffer((4,), "int32")\n    A[0] = 1',
    ],
)
def test_source_store_uses_public_builder_and_propagates_errors(monkeypatch, error_type, body):
    error = error_type("store rejected")
    calls = []

    def reject_store(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(tir_builder, "buffer_store", reject_store)
    source = f"@T.prim_func\ndef main():\n    {body}\n"
    with pytest.raises(tvm.error.DiagnosticError, match="store rejected") as raised:
        parser.parse(source)
    assert raised.value.__cause__ is error
    assert len(calls) == 1


def test_explicit_relax_emit_retains_source_binding_api():
    source = """
@R.function
def main():
    x = R.emit(R.const(1))
    return x
"""
    with IRBuilder() as context:
        with R.function():
            R.func_name("main")
            value = native_R.emit(R.const(1))
            bound = native_R.emit(value)
            R.func_ret_value(bound)
    ir.assert_structural_equal(context.get(), parser.parse(source))


def test_emit_keeps_existing_buffer_store_identity_and_span():
    with IRBuilder() as builder:
        with T.function():
            T.func_name("store")
            buffer = tirx.decl_buffer((4,), "int32")
            statement = PS.at(("store.py", 3, 3, 1, 12), tirx.BufferStore(buffer, 1, [0]))
            T.emit(statement)
    stored = builder.get().body
    assert stored.same_as(statement)
    assert stored.span.line == 3


@pytest.mark.parametrize("dialect", ["tirx", "relax"])
def test_generated_statements_use_dialect_emit_protocol(monkeypatch, dialect):
    builder = tir_builder if dialect == "tirx" else R
    original = builder.emit_
    calls = []

    def consume(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(builder, "emit_", consume)
    source = (
        "@T.prim_func\ndef main():\n    T.evaluate(1)\n"
        if dialect == "tirx"
        else (
            "@R.function(pure=False)\ndef main(x: R.Tensor((4,), 'float32')):\n"
            "    R.print(x)\n    return x\n"
        )
    )
    function = parser.parse(source)
    assert len(calls) == 1
    if dialect == "tirx":
        assert isinstance(function.body, tirx.Evaluate)
        assert function.body.span.line == 3
    else:
        binding = function.body.blocks[0].bindings[0]
        assert binding.value.op.name == "relax.print"
        assert binding.value.span.line == 3
