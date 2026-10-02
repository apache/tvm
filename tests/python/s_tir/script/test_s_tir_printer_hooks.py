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
"""Dialect-owned S-TIR printer hooks preserve representation and round trips."""

import pytest

import tvm
from tvm import s_tir, tirx
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def _roundtrip(obj):
    source = obj.script()
    reparsed = tvm.script.from_source(source, extra_vars={"I": I, "T": T, "Ts": Ts})
    tvm.ir.assert_structural_equal(obj, reparsed)
    return source


@pytest.mark.parametrize("realize", [False, True])
def test_standalone_block_repr(realize):
    block = s_tir.SBlock([], [], [], "block", tirx.Evaluate(0))
    node = s_tir.SBlockRealize([], True, block) if realize else block
    source = node.script()
    assert 'Ts.sblock("block"' in source
    assert repr(node) == source
    assert str(node) == source


def test_standalone_match_buffer_repr():
    source = tirx.decl_buffer((16,), "float32", name="source")
    target = tirx.decl_buffer((8,), "float32", name="target")
    match = s_tir.MatchBufferRegion(
        target, tirx.BufferRegion(source, [tvm.ir.Range.from_min_extent(4, 8)])
    )
    script = match.script()
    assert "Ts.match_buffer(source[4:12]" in script
    assert repr(match) == script
    assert str(match) == script


def test_match_buffer_in_block_roundtrip():
    @Ts.prim_func
    def main(A: T.Buffer((16,), "float32")):
        with Ts.sblock("block"):
            B = Ts.match_buffer(A[4:12], (8,), "float32")
            Ts.reads(A[4])
            Ts.writes()
            T.evaluate(B[0])

    source = _roundtrip(main)
    assert source.count("Ts.match_buffer(") == 1


@pytest.mark.parametrize("stir_first", [False, True])
def test_mixed_function_layout_defaults(stir_first):
    a = tirx.decl_buffer((16,), "float32", name="a", layout=None)
    b = tirx.decl_buffer((16,), "float32", name="b", layout=None)
    stir = tirx.PrimFunc([a], tirx.Evaluate(a[0])).with_attr("s_tir", True)
    plain = tirx.PrimFunc([b], tirx.Evaluate(b[0]))
    functions = {"a": stir, "b": plain} if stir_first else {"a": plain, "b": stir}
    module = tvm.IRModule(functions)
    source = _roundtrip(module)
    assert source.count("layout=None") == 1
    assert "@Ts.prim_func" in source
    assert "@T.prim_func" in source
    assert '"s_tir"' not in source
    assert "layout=None" in plain.script()
    assert "layout=None" not in stir.script()


@pytest.mark.parametrize(
    "raw_kind", ["canonical", "pointer_destination", "buffer_source", "result_type", "attrs"]
)
def test_ldg32_roundtrip(raw_kind):
    source = tirx.decl_buffer((1,), "float32", name="source")
    target = tirx.decl_buffer((1,), "float32", name="target")
    destination = target.data if raw_kind == "pointer_destination" else target[0]
    call = tvm.ir.Call(
        "tirx.s_tir.ldg32",
        [
            destination,
            tirx.const(1, "int32"),
            source if raw_kind == "buffer_source" else source[0],
            tirx.const(0, "int32"),
        ],
        attrs=tvm.ir.DictAttrs({"tag": 1}) if raw_kind == "attrs" else None,
        ret_ty=tvm.ir.PrimType("float16" if raw_kind == "result_type" else "float32"),
    )
    function = tirx.PrimFunc([source, target], tirx.Evaluate(call))
    printed = _roundtrip(function)
    if raw_kind == "canonical":
        assert "T.s_tir.ldg32(" in printed
    else:
        assert "I.Call(" in printed
        assert "T.s_tir.ldg32(" not in printed


@pytest.mark.parametrize("raw_kind", ["canonical", "extra_operand", "attrs"])
def test_cp_async_raw_roundtrip(raw_kind):
    source = tirx.decl_buffer((16,), "float16", name="source")
    target = tirx.decl_buffer((16,), "float16", name="target")
    operands = [
        target.data,
        tirx.const(0, "int32"),
        source.data,
        tirx.const(0, "int32"),
        tirx.const(16, "int32"),
    ]
    if raw_kind == "extra_operand":
        operands.append(tirx.const(1, "int32"))
    call = tvm.ir.Call(
        "tirx.s_tir.cp_async_raw",
        operands,
        attrs=tvm.ir.DictAttrs({"tag": 1}) if raw_kind == "attrs" else None,
        ret_ty=tvm.ir.PrimType("float16"),
    )
    function = tirx.PrimFunc([source, target], tirx.Evaluate(call))
    printed = _roundtrip(function)
    if raw_kind == "canonical":
        assert 'T.s_tir.cp_async_raw("float16"' in printed
    else:
        assert "I.Call(" in printed
        assert "T.s_tir.cp_async_raw(" not in printed
