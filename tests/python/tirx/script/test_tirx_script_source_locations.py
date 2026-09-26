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

"""TIRx script source locations."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import tvm_ffi
from tvm_ffi import structural_walk

import tvm
import tvm.testing
from tvm import ir
from tvm.ir import Call, SequentialSpan, TensorLoad, assert_structural_equal, prim
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.ir_builder.base import AlreadyEmitted
from tvm.script.parser.inspect_source import Source
from tvm.script.tirx import tile as Tx
from tvm.tirx.stmt import TilePrimitiveCall


def test_parser_attaches_span_to_direct_call():
    sources = []

    @T.prim_func
    @_capture_source(sources)
    def direct_call():
        T.device_entry()
        barriers = T.alloc_buffer((1,), "uint64", scope="shared")
        T.cuda.mbarrier_wait(
            T.address_of(barriers[0]),
            0,
        )

    source = sources[0]
    call_ast = source.as_ast().body[0].body[-1].value
    func = direct_call
    call = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, Call) and getattr(node.op, "name", None) == "tirx.cuda.mbarrier_wait"
        ),
    )

    assert _span_range(call.span) == _span_range(source.to_span(call_ast))


def _capture_source(sources):
    """Keep original source coordinates before a definition-site construction."""

    def capture(function):
        sources.append(Source(function))
        return function

    return capture


def _span_range(span):
    return (
        span.source_name.name,
        span.line,
        span.column,
        span.end_line,
        span.end_column,
    )


def _find_ir_node(func, predicate):
    nodes = []
    structural_walk(func.body, nodes.append, order="post")
    matches = [node for node in nodes if predicate(node)]
    assert len(matches) == 1
    return matches[0]


def test_parser_attaches_span_to_nested_tensor_load():
    sources = []

    @T.prim_func
    @_capture_source(sources)
    def nested_load():
        source_buffer = T.alloc_buffer((1,), "int32")
        output = T.alloc_buffer((1,), "int32")
        output[0] = source_buffer[0] + 1

    source = sources[0]
    load_ast = source.as_ast().body[0].body[-1].value.left
    func = nested_load
    load = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, TensorLoad) and getattr(node.source, "name", None) == "source_buffer"
        ),
    )

    assert _span_range(load.span) == _span_range(source.to_span(load_ast))


def test_parser_retains_inline_call_site_and_definition_spans():
    @T.inline
    def wait_impl(barrier):
        T.cuda.mbarrier_wait(barrier, 0)

    wait_source = Source(wait_impl.__wrapped__)
    wait_call_ast = wait_source.as_ast().body[0].body[0].value
    wait = wait_impl

    sources = []

    @T.prim_func
    @_capture_source(sources)
    def inline_call():
        T.device_entry()
        barriers = T.alloc_buffer((1,), "uint64", scope="shared")
        wait(T.address_of(barriers[0]))

    caller_source = sources[0]
    caller_call_ast = caller_source.as_ast().body[0].body[-1].value
    func = inline_call
    call = _find_ir_node(
        func,
        lambda node: (
            isinstance(node, Call) and getattr(node.op, "name", None) == "tirx.cuda.mbarrier_wait"
        ),
    )

    assert isinstance(call.span, SequentialSpan)
    assert [_span_range(span) for span in call.span.spans] == [
        _span_range(caller_source.to_span(caller_call_ast)),
        _span_range(wait_source.to_span(wait_call_ast)),
    ]


def test_parser_attaches_span_to_tile_primitive_call():
    sources = []

    @T.prim_func
    @_capture_source(sources)
    def tile_call():
        A = T.alloc_buffer((16,), "float32")
        Tx.memset(A[0:16], T.float32(0))

    source = sources[0]
    call_ast = source.as_ast().body[0].body[-1].value
    func = tile_call
    call = _find_ir_node(func, lambda node: isinstance(node, TilePrimitiveCall))

    assert _span_range(call.span) == _span_range(source.to_span(call_ast))


def test_parser_spans_do_not_affect_structural_identity():
    source_a = """@T.prim_func\ndef f():\n    T.evaluate(1)\n"""
    source_b = """\n\n@T.prim_func\ndef f():\n    T.evaluate(1)\n"""

    func_a = tvm.script.from_source(source_a, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})
    func_b = tvm.script.from_source(source_b, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})

    assert _span_range(func_a.body.span) == ("<str>", 3, 5, 3, 18)
    assert _span_range(func_b.body.span) == ("<str>", 5, 5, 5, 18)
    assert tvm_ffi.structural_hash(func_a) == tvm_ffi.structural_hash(func_b)
    assert_structural_equal(func_a, func_b)


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


def _line_of(function, statement):
    lines, first = inspect.getsourcelines(function)
    return first + next(index for index, line in enumerate(lines) if line.strip() == statement)


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


def _position(test, statement, expression):
    lines, start = inspect.getsourcelines(test)
    index, line = next((i, line) for i, line in enumerate(lines) if line.strip() == statement)
    column = line.index(expression) + 1
    return start + index, column, start + index, column + len(expression)


def _span_position(span):
    return span.line, span.column, span.end_line, span.end_column
