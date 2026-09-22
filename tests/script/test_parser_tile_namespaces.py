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
"""Shared tile namespaces emit native calls through the new builder."""

import pytest

from tvm import ir, tirx
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.tirx import script as T
from tvm.tirx.script import tile

SOURCE = """
@T.prim_func
def main(A: T.Buffer((1,), "float32"), B: T.Buffer((1,), "float32")):
    CALL(B, A)
"""


def environment():
    return {
        "T": T,
        "Tx": tile,
        "copy": tile.copy,
        "warp_copy": tile.warp.copy,
        "warp": tile.warp,
    }


def assert_copy(function, scope):
    call = function.body
    assert isinstance(call, tirx.TilePrimitiveCall)
    assert call.op.name == "tirx.tile.copy"
    assert call.scope.name == scope
    assert call.span is not None
    assert len(call.args) == 2
    for argument, parameter in zip(call.args, reversed(function.params)):
        buffer = parameter
        assert argument.source.same_as(buffer)
        ir.assert_structural_equal(argument, buffer[:])
    assert not call.workspace
    assert not call.config


@pytest.mark.parametrize(
    "call,scope",
    [
        ("Tx.copy", "thread"),
        ("Tx.thread.copy", "thread"),
        ("Tx.warp.copy", "warp"),
        ("Tx.wg.copy", "warpgroup"),
        ("Tx.warpgroup.copy", "warpgroup"),
        ("Tx.cta.copy", "cta"),
        ("Tx.cluster.copy", "cluster"),
        ("T.warp.copy", "warp"),
        ("T.tile.copy", "thread"),
        ("copy", "thread"),
        ("warp_copy", "warp"),
        ("warp.copy", "warp"),
    ],
)
def test_tile_module_scope_and_captured_operations(call, scope):
    result = parser.parse(SOURCE.replace("CALL", call), extra_vars=environment())
    assert_copy(result, scope)


@pytest.mark.parametrize(
    "imports,call,scope",
    [
        ("from tvm.tirx.script import tile as Tx", "Tx.copy", "thread"),
        ("from tvm.tirx.script.tile import copy", "copy", "thread"),
        ("from tvm.tirx.script.tile import warp", "warp.copy", "warp"),
    ],
)
def test_source_imports_use_shared_tile_operations(imports, call, scope):
    result = parser.parse(imports + "\n" + SOURCE.replace("CALL", call), extra_vars={"T": T})
    assert_copy(result, scope)


@pytest.mark.parametrize("name", ["copy", "cast", "max", "min", "sqrt"])
@pytest.mark.parametrize("scope", [None, "warp"])
def test_tile_only_validation(name, scope):
    namespace = tile if scope is None else getattr(tile, scope)
    with pytest.raises(TypeError, match="tile-only and expects `dst`"):
        getattr(namespace, name)(1, 2)


@pytest.mark.parametrize("name", ["cast", "max", "min", "sqrt"])
def test_tile_only_source_validation(name):
    buffer = tirx.decl_buffer((1,), "float32")
    with pytest.raises(TypeError, match="tile-only and expects `src`"):
        getattr(tile.warp, name)(buffer, 1)


def test_captured_bound_operation_keeps_exact_scope_identity():
    scope = tirx.ExecScope("warp")
    namespace = tile.warp.__class__(scope, "test_warp")
    captured = namespace.copy
    buffer = tirx.decl_buffer((1,), "float32")
    with IRBuilder() as builder:
        assert captured(buffer, buffer) is None
    call = builder.get()
    assert call.scope.same_as(scope)
    assert all(argument.source.same_as(buffer) for argument in call.args)


def test_opaque_helper_uses_shared_tile_namespace():
    def helper(dst, src):
        tile.warp.copy(dst, src)

    result = parser.parse(SOURCE.replace("CALL", "helper"), extra_vars={"T": T, "helper": helper})
    assert_copy(result, "warp")


def test_scope_namespaces_reject_non_tile_attributes():
    assert not hasattr(tile.warp, "compose_op")
    assert not hasattr(tile.warp, "_fn")


def test_gemm_captured_namespace_preserves_operand_order():
    source = """
@T.prim_func
def main(
    A: T.Buffer((1, 1), "float32"),
    B: T.Buffer((1, 1), "float32"),
    C: T.Buffer((1, 1), "float32"),
):
    Tx.gemm(C, A, B, C)
"""
    result = parser.parse(source, extra_vars=environment())
    call = result.body
    assert call.op.name == "tirx.tile.gemm"
    assert call.scope.name == "thread"
    assert len(call.args) == 8
    assert not bool(call.args[4])
    assert not bool(call.args[5])
    assert call.args[6].value == 1.0
    assert call.args[7].value == 0.0
    for argument, index in zip(call.args, [2, 0, 1, 2]):
        buffer = result.params[index]
        assert argument.source.same_as(buffer)
        ir.assert_structural_equal(argument, buffer[:, :])
