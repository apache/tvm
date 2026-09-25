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
"""TIRX parser integration for source locations."""

from __future__ import annotations

# Script-local bindings are inspected through the constructed IR.
import inspect
from types import SimpleNamespace

import pytest

from tvm import ir
from tvm.ir import prim
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.ir_builder.base import AlreadyEmitted


def _line_of(function, statement):
    lines, first = inspect.getsourcelines(function)
    return first + next(index for index, line in enumerate(lines) if line.strip() == statement)


def _position(test, statement, expression):
    lines, start = inspect.getsourcelines(test)
    index, line = next((i, line) for i, line in enumerate(lines) if line.strip() == statement)
    column = line.index(expression) + 1
    return start + index, column, start + index, column + len(expression)


def _span_position(span):
    return span.line, span.column, span.end_line, span.end_column


def test_statement_receipts_keep_emitted_nodes_and_spans():
    # Direct statement hooks return the stored nodes, with no extra emit or duplicate statement.
    location = ir.Span(ir.SourceName("direct_builder.py"), 5, 5, 3, 17)
    with I.IRBuilder():
        with T.function_(private=True) as frame:
            T.func_name_("receipts")
            output = T.arg_("output", T.Buffer((1,), "int32"))
            stored = T.setitem_(value=3, target=output, key=0, span=location)
            holder = SimpleNamespace(value=output)
            updated = T.setattr_(holder, "value", 4, span=location)
            with T.while_(T.bool(True)):
                continued = T.continue_(span=location)
                broken = T.break_(span=location)
            returned = T.return_(T.int32(7), span=location)

    body = frame.function.body.seq
    assert len(body) == 4 and len(body[2].body.seq) == 2
    receipts = [stored, updated, continued, broken, returned]
    statements = [body[0], body[1], *body[2].body.seq, body[3]]
    for receipt, statement in zip(receipts, statements):
        assert isinstance(receipt, AlreadyEmitted)
        assert receipt.value.same_as(statement)
        assert statement.span.same_as(location)


def test_native_bind_keeps_returned_and_stored_variable_identity():
    # Aliases of an explicitly produced native variable must not rename, respan or bind it again.
    from tvm import ir, tirx

    span = ir.Span(ir.SourceName("producer.py"), 7, 7, 2, 19)
    produced = ir.Var("producer_name", "int32", span)
    calls, observed = [], []

    def argument(value):
        calls.append(value)
        return value

    def observe(value):
        observed.append(value)
        assert value.same_as(produced) and value.span.same_as(span)
        assert value.name == "producer_name"

    @T.prim_func
    def main(x: T.int32):
        renamed = T.bind(argument(x), var=produced)
        alias = renamed
        called = argument(alias)
        observe(renamed)
        observe(alias)
        observe(called)
        T.evaluate(called)

    binding, use = main.body.seq
    assert isinstance(binding, tirx.Bind) and isinstance(use, tirx.Evaluate)
    assert binding.var.same_as(produced) and use.value.same_as(produced)
    assert len(observed) == 3 and len(calls) == 2
    assert calls[0].same_as(main.params[0]) and calls[1].same_as(produced)


def test_native_view_keeps_producer_identity_name_and_span(monkeypatch):
    # A native view must preserve its producer name/span and declare its storage exactly once.
    from functools import wraps

    from tvm import ir, tirx
    from tvm.script.ir_builder import base

    captured = T.Buffer((4, 4), "float32")
    original = type(captured).view
    seen, produced, observed = [], [], []
    span = ir.Span(ir.SourceName("producer.py"), 7, 7, 2, 19)

    @wraps(original)
    def view(buffer, *args):
        seen.append("view")
        value = base.at_(span, original(buffer, *args))
        produced.append((value, value.name, value.span))
        return value

    def mark():
        seen.append("argument")
        return 16

    def observe(value):
        observed.append((value, value.span))

    monkeypatch.setattr(type(captured), "view", view)

    @T.prim_func
    def main(A: captured):
        renamed = captured.view(mark())
        alias = renamed
        observe(renamed)
        observe(alias)
        alias[0] = 0
        captured.view(mark())

    assert seen == ["argument", "view", "argument", "view"]
    value, name, produced_span = produced[0]
    assert len(produced) == len(observed) == 2
    assert name != "renamed" and value.name == name
    assert all(
        item.same_as(value) and location.same_as(produced_span) for item, location in observed
    )
    nodes = list(main.body.seq)
    assert len(nodes) == 3 and not any(isinstance(node, tirx.Bind) for node in nodes)
    assert nodes[0].buffer.same_as(value) and nodes[1].buffer.same_as(value)
    assert nodes[2].buffer.same_as(produced[1][0])
    ir.assert_structural_equal(nodes[0].data, captured.data)
    ir.assert_structural_equal(nodes[2].data, captured.data)


def test_native_binding_preserves_metadata_but_binds_buffer_expressions():
    # Inert metadata must retain resource identity; a non-Var buffer expression still needs a
    # located binding.
    from tvm import ir, tirx
    from tvm.script.ir_builder import base

    producer_span = ir.Span(ir.SourceName("producer.py"), 7, 7, 2, 19)
    buffer = T.Buffer((4,), "float32")
    base.at_(producer_span, buffer)
    layout = T.TileLayout(T.S[4])

    @T.meta_class
    class Holder:
        def __init__(self, resource):
            self.resource = resource

    holder = Holder(buffer)
    projection = ir.TupleGetItem(ir.Tuple([buffer]), 0)
    values = [layout, holder, projection]
    calls, observed, resource_spans = [], [], []

    def make(index):
        resource_spans.append(buffer.span)
        calls.append(index)
        return values[index]

    def observe(*items):
        observed.extend(items)

    @T.prim_func
    def main(A: buffer):
        renamed_layout = make(0)
        renamed_holder = make(1)
        bound = make(2)
        observe(renamed_layout, renamed_holder)
        make(0)
        make(1)

    assert calls == [0, 1, 2, 0, 1]
    assert observed[0].same_as(layout)
    assert observed[1] is holder and holder.resource.same_as(buffer)
    assert all(buffer.span.same_as(span) for span in resource_spans)
    binding = main.body
    assert isinstance(binding, tirx.Bind) and binding.value.same_as(projection)
    assert binding.var.name == "bound"
    assert tirx.is_buffer_var(binding.var)
    assert not isinstance(projection, ir.Var)
    line = _line_of(
        test_native_binding_preserves_metadata_but_binds_buffer_expressions, "bound = make(2)"
    )
    assert (binding.var.span.line, binding.var.span.column, binding.var.span.end_column) == (
        line,
        9,
        14,
    )
    assert (projection.span.line, projection.span.column, projection.span.end_column) == (
        line,
        17,
        24,
    )
    with pytest.raises(TypeError):

        @T.prim_func
        def invalid():
            object()


def test_non_call_expression_reads_keep_their_source_range():
    # Bare name, property, item and literal emissions must retain their written
    # ranges in actual IR, with each host read evaluated once.
    value = prim.IntImm("int32", 1)
    attribute = prim.IntImm("int32", 2)
    item = prim.IntImm("int32", 3)
    reads = []

    class Holder:
        @property
        def value(self):
            reads.append("attribute")
            return attribute

        def __getitem__(self, index):
            reads.append(index)
            return item

    holder = Holder()

    @T.prim_func
    def main():
        value
        holder.value
        holder[0]
        7

    nodes = list(main.body.seq)
    assert [int(node.value) for node in nodes] == [1, 2, 3, 7]
    assert nodes[0].value.same_as(value)
    assert nodes[1].value.same_as(attribute)
    assert nodes[2].value.same_as(item)
    assert reads == ["attribute", 0]
    for node, expression in zip(nodes, ("value", "holder.value", "holder[0]", "7")):
        expected = _position(
            test_non_call_expression_reads_keep_their_source_range, expression, expression
        )
        assert _span_position(node.span) == expected
        assert node.span.source_name.name == __file__
