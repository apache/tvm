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
"""Source context belongs to the actual construction result."""

import pytest

from tvm import ir, tirx
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import parser_support as PS
from tvm.script.ir_builder.base import BypassEmit
from tvm.script import tirx as T
from tvm.tirx.script.builder import ir as native


def loc(line, name="source.py", column=1, end_column=20):
    return (name, line, line, column, end_column)


def locations(value):
    span = value.span
    spans = span.spans if isinstance(span, ir.SequentialSpan) else [span]
    return [(str(s.source_name.name), s.line, s.column, s.end_column) for s in spans]


def test_evaluate_receipt_annotates_stored_statement():
    with IRBuilder() as builder:
        receipt = T.evaluate(1)
        assert isinstance(receipt, BypassEmit)
        assert PS.at(loc(3), receipt) is receipt
        T.emit(receipt)
    result = builder.get()
    assert isinstance(result, tirx.Evaluate)
    assert result.same_as(receipt.stmt)
    assert locations(result) == [("source.py", 3, 1, 20)]


def test_emit_ordinary_value_once():
    with IRBuilder() as builder:
        with T.function():
            T.func_name("single")
            value = PS.at(loc(4), tirx.IntImm("int32", 2))
            T.emit(value)
    assert isinstance(builder.get().body, tirx.Evaluate)


def test_at_preserves_native_expression_identity():
    with IRBuilder():
        value = tirx.IntImm("int32", 2)
        assert PS.at(loc(4), value) is value
        assert locations(value) == [("source.py", 4, 1, 20)]


@pytest.mark.parametrize("value", [None, 3, "python", object()])
def test_at_preserves_python_values(value):
    with IRBuilder():
        assert PS.at(loc(5), value) is value


def test_native_frame_retains_span_until_exit():
    with IRBuilder() as builder:
        frame = native.serial(0, 4)
        assert PS.at(loc(6), frame) is frame
        with frame:
            receipt = PS.with_at_scope(loc(7), lambda: T.evaluate(1))
    result = builder.get()
    assert isinstance(result, tirx.For)
    assert locations(result) == [("source.py", 6, 1, 20)]
    assert result.body.same_as(receipt.stmt)
    assert locations(result.body) == [("source.py", 7, 1, 20)]


def test_wrapped_frame_retains_span_until_exit():
    with IRBuilder() as builder:
        frame = T.While(tirx.IntImm("bool", 1))
        assert PS.at(loc(8), frame) is frame
        with frame:
            T.evaluate(0)
    assert isinstance(builder.get(), tirx.While)
    assert locations(builder.get()) == [("source.py", 8, 1, 20)]


def test_at_retains_definition_after_caller_scope_returns():
    with IRBuilder() as builder:

        def helper():
            return PS.with_at_scope(loc(2, "definition.py"), lambda: T.evaluate(1))

        receipt = PS.with_at_scope(loc(9, "caller.py"), helper)
        assert PS.at(loc(9, "caller.py"), receipt) is receipt
        T.emit(receipt)
    assert locations(builder.get()) == [("caller.py", 9, 1, 20), ("definition.py", 2, 1, 20)]


def test_enclosing_locations_collapse_without_losing_definition():
    with IRBuilder():
        value = tirx.IntImm("int32", 1)
        PS.at(loc(10, column=5, end_column=10), value)
        PS.at(loc(10), value)
        assert locations(value) == [("source.py", 10, 5, 10)]
        PS.at(loc(10, column=6, end_column=8), value)
        assert locations(value) == [("source.py", 10, 6, 8)]


def test_nested_frame_context_has_no_repeated_caller():
    with IRBuilder() as builder:
        with builder.with_source_span(ir.Span(ir.SourceName("caller.py"), 1, 1, 1, 20)):
            frame = PS.with_at_scope(loc(2, "definition.py"), lambda: native.serial(0, 2))
            with frame:
                T.evaluate(0)
    assert locations(builder.get()) == [("caller.py", 1, 1, 20), ("definition.py", 2, 1, 20)]


def test_composition_merges_shared_prefix_of_distinct_definition_chains():
    with IRBuilder():
        caller = ir.Span(ir.SourceName("caller.py"), 1, 1, 1, 20)
        original = ir.Span(ir.SourceName("definition.py"), 2, 2, 1, 20)
        value = tirx.IntImm("int32", 1)
        PS.at(ir.SequentialSpan([caller, original]), value)
        with IRBuilder.current().with_source_span(caller):
            PS.at(loc(3, "definition.py"), value)
    assert locations(value) == [
        ("caller.py", 1, 1, 20),
        ("definition.py", 3, 1, 20),
        ("definition.py", 2, 1, 20),
    ]


def test_tile_insertion_composes_caller_on_existing_statement_span():
    from tvm.tirx import operator as tile_operator
    from tvm.tirx.script.builder import tirx as tile_builder

    buffer = tirx.decl_buffer((1,), "int32")
    with IRBuilder() as builder:
        statement = PS.at(loc(2, "definition.py"), tile_operator.Zero(buffer[:], buffer[:]))
        result = PS.with_at_scope(loc(9, "caller.py"), lambda: tile_builder.f_insert(statement))
        assert result is None
        T.emit(result)
    assert builder.get().same_as(statement)
    assert locations(builder.get()) == [
        ("caller.py", 9, 1, 20),
        ("definition.py", 2, 1, 20),
    ]
