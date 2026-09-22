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
"""Scope IDs retain the native variables defined by their emitted statements."""

import pytest

import tvm
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder.base import BypassBind, at
from tvm.script.ir_builder.type_var_frame import TypeVarFrame
from tvm.script import tirx as T


@pytest.mark.parametrize(
    "constructor, target, extents, names",
    [
        ("thread_id", "tx", "[32]", ["tx"]),
        ("cta_id", "bx, by", "[2, 3]", ["bx", "by"]),
        ("warp_id", "wx", "[4]", ["wx"]),
    ],
)
def test_scope_id_declarations_and_uses_share_identity(constructor, target, extents, names):
    # Tuple dimensions are each used, not just the leading scope variable.
    evaluations = "\n".join(f"        T.evaluate({name})" for name in names)
    source = f"""
@T.prim_func
def main():
    T.device_entry()
    {target} = T.{constructor}({extents})
    if {names[0]} == 0:
{evaluations}
"""
    with IRBuilder() as builder:
        with T.function():
            T.func_name("main")
            T.device_entry()
            declaration = getattr(T, constructor)(eval(extents)).value
            variables = declaration if isinstance(declaration, tuple) else (declaration,)
            with T.If(tvm.tirx.EQ(variables[0], 0)):
                with T.Then():
                    for variable in variables:
                        T.evaluate(variable)
    expected = builder.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    declaration, branch = actual.body.body.seq
    variables = getattr(declaration, "def").def_ids
    assert variables[0].same_as(branch.condition.a)
    statements = (
        branch.then_case.seq
        if isinstance(branch.then_case, tvm.tirx.SeqStmt)
        else [branch.then_case]
    )
    for variable, statement in zip(variables, statements):
        assert variable.same_as(statement.value)


def test_scope_owned_variable_does_not_replace_signature_symbol():
    with IRBuilder(), TypeVarFrame() as symbols:
        signature_symbol = symbols.resolve("tx", "int32")
        with T.function():
            T.device_entry()
            variable = T.thread_id([32])
            bound = T.bind_(variable, name="tx")
            assert bound.same_as(variable.value)
            assert not bound.same_as(signature_symbol)
            assert symbols.resolve("tx").same_as(signature_symbol)
            T.evaluate(bound)


def test_unowned_anonymous_declarations_still_reuse_function_symbols():
    with IRBuilder(), TypeVarFrame() as symbols:
        signature_symbol = symbols.resolve("n", "int32")
        with T.function():
            value = T.int32()
            assert not value.same_as(signature_symbol)
            assert T.bind_(value, name="n").same_as(signature_symbol)
            T.evaluate(0)


@pytest.mark.parametrize("dialect", ["tirx", "relax"])
def test_bypass_bind_returns_exact_value_before_any_binding_policy(dialect):
    from importlib import import_module

    builder = import_module(f"tvm.{dialect}.script.builder")
    value = object()
    # No builder or symbol frame is active. Neither an annotation nor a previous
    # binding may cause ordinary binding logic to inspect the wrapped value.
    assert (
        builder.bind_(BypassBind(value), name="x", ty=object(), previous=object(), declaration=True)
        is value
    )


def test_scope_tuple_assignment_returns_native_values():
    with IRBuilder(), TypeVarFrame():
        with T.function():
            T.device_entry()
            result = T.cta_id([2, 3])
            assert isinstance(result, BypassBind)
            values = T.bind_(result, name="ids")
            assert values is result.value
            unpacked = T.unpack(result)
            for variable, item in zip(values, unpacked):
                assert isinstance(item, BypassBind)
                assert T.bind_(item, name="axis").same_as(variable)
                T.evaluate(variable)


@pytest.mark.parametrize("count", [1, 2])
def test_bypass_source_attachment_preserves_value_identity(count):
    span = tvm.ir.Span(tvm.ir.SourceName("scope.py"), 3, 3, 4, 25)
    with IRBuilder():
        values = tuple(tvm.ir.Var("", "int32") for _ in range(count))
        result = BypassBind(values[0] if count == 1 else values)
        assert at(span, result) is result
        for value in values:
            assert value.span.same_as(span)


@pytest.mark.parametrize("track_span", [True, False])
def test_scope_ids_in_single_tuple_target(track_span):
    source = """
@T.prim_func
def main():
    T.device_entry()
    ids = T.cta_id([2, 3])
    T.evaluate(ids[0])
    T.evaluate(ids[1])
"""
    actual = parser.parse(source, track_span=track_span)
    declaration, first, second = actual.body.body.seq
    variables = getattr(declaration, "def").def_ids
    assert variables[0].same_as(first.value)
    assert variables[1].same_as(second.value)
    if track_span:
        assert variables[0].span is not None
        assert variables[1].span is not None


@pytest.mark.parametrize(
    "constructor,args",
    [
        ("scope_id", ([32], "cta", "thread")),
        ("cluster_id", ([2],)),
        ("cta_id", ([2],)),
        ("cta_id_in_cluster", ([2],)),
        ("cta_id_in_pair", ()),
        ("warpgroup_id", ([2],)),
        ("warp_id", ([4],)),
        ("warp_id_in_wg", ([4],)),
        ("lane_id", ([32],)),
        ("thread_id", ([32],)),
        ("thread_id_in_wg", ([128],)),
    ],
)
def test_all_scope_id_helpers_opt_into_bypass(constructor, args):
    with IRBuilder() as builder, TypeVarFrame():
        with T.function():
            T.device_entry()
            result = getattr(T, constructor)(*args)
            assert isinstance(result, BypassBind)
            value = T.bind_(result, name="id")
            T.evaluate(value)
        function = builder.get()
    declaration, use = function.body.body.seq
    assert getattr(declaration, "def").def_ids[0].same_as(use.value)


@pytest.mark.parametrize("extents", [[32], [2, 3]])
@pytest.mark.parametrize("emitter", [T.emit, T.emit_])
def test_standalone_scope_ids_direct(extents, emitter):
    with IRBuilder() as builder, TypeVarFrame():
        with T.function():
            T.device_entry()
            result = T.cta_id(extents)
            emitter(result)
        function = builder.get()
    declaration = function.body.body
    assert isinstance(declaration, tvm.tirx.ScopeIdDefStmt)
    values = result.value if isinstance(result.value, tuple) else (result.value,)
    for declared, value in zip(getattr(declaration, "def").def_ids, values):
        assert declared.same_as(value)


@pytest.mark.parametrize("extents", [[32], [2, 3]])
@pytest.mark.parametrize("track_span", [True, False])
def test_standalone_scope_ids_source(extents, track_span):
    source = f"""
@T.prim_func
def main():
    T.device_entry()
    T.cta_id({extents!r})
"""
    with IRBuilder() as builder:
        with T.function():
            T.func_name("main")
            T.device_entry()
            T.cta_id(extents)
    expected = builder.get()
    actual = parser.parse(source, track_span=track_span)
    tvm.ir.assert_structural_equal(expected, actual)
    assert isinstance(actual.body.body, tvm.tirx.ScopeIdDefStmt)


@pytest.mark.parametrize("emitter", [T.emit, T.emit_])
@pytest.mark.parametrize("tuple_value", [False, True])
def test_binding_bypass_does_not_suppress_value_emission(emitter, tuple_value):
    with IRBuilder() as builder, TypeVarFrame():
        with T.function():
            values = (tvm.tirx.IntImm("int32", 7), tvm.tirx.IntImm("int32", 9))
            emitter(BypassBind(values if tuple_value else values[0]))
        function = builder.get()
    statements = function.body.seq if tuple_value else [function.body]
    assert [statement.value.value for statement in statements] == ([7, 9] if tuple_value else [7])
