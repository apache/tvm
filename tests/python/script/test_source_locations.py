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
"""Generated IR retains the user's call sites and definition locations.
Fileless documentation execution also recovers the correct decorated class source.
"""

from __future__ import annotations

# Script-local assignments are observed through the constructed IR.
# ruff: noqa: F841
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
from minilang import Value

from tvm import ir
from tvm.ir import prim
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.ir_builder.base import AlreadyEmitted


def span_lines(value):
    spans = value.span.spans if isinstance(value.span, ir.SequentialSpan) else [value.span]
    return [span.line for span in spans]


@pytest.fixture
def spanned(spanned_language):
    spanned_language.M.node = lambda value: prim.IntImm("int32", value)
    return spanned_language


def source_line(test, fragment):
    lines, start = inspect.getsourcelines(test)
    return start + next(i for i, line in enumerate(lines) if line.strip() == fragment)


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


def test_module_source_calls_have_context_before_annotations(spanned):
    # Module construction calls must receive the class-body source context, including calls
    # before member annotations.
    M = spanned.M
    seen = []

    def record():
        node = prim.IntImm("int32", 3)
        seen.append(node)
        return node

    @I.ir_module(extra_vars={"record": record})
    class Module:
        record()

        @M.function
        def main():
            M.record(1)

    assert "main" in Module
    # The ordinary Python class body runs before the decorator; only the parsed
    # class call has a source range attached to its returned expression.
    assert len(seen) == 2
    assert seen[0].span is None
    assert seen[1].span is not None
    assert seen[1].span.line == source_line(
        test_module_source_calls_have_context_before_annotations, "record()"
    )


def test_source_inline_keeps_caller_and_each_definition_location(spanned):
    # A macro emitting twice must retain both original definition lines under the caller,
    # without reusing the first emission span.
    M = spanned.M

    @M.inline
    def inner(x):
        M.record(M.node(x))
        M.record(M.node(x + 1))

    @M.function
    def main():
        inner(1)

    statements = [value for _, value in main.body if isinstance(value, prim.IntImm)]
    assert len(statements) == 2
    assert [int(value) for value in statements] == [1, 2]
    test = test_source_inline_keeps_caller_and_each_definition_location
    caller = source_line(test, "inner(1)")
    assert span_lines(statements[0]) == [caller, source_line(test, "M.record(M.node(x))")]
    assert span_lines(statements[1]) == [caller, source_line(test, "M.record(M.node(x + 1))")]
    assert all(
        str(span.source_name.name) == __file__ for value in statements for span in value.span.spans
    )


@pytest.fixture
def calls(spanned_language):
    M = spanned_language.M
    M.evaluate = lambda value: value
    M.call_extern = lambda dtype, name: ir.Call(ir.GlobalVar(name), [], ret_ty=dtype)
    return M


def _position(test, statement, expression):
    lines, start = inspect.getsourcelines(test)
    index, line = next((i, line) for i, line in enumerate(lines) if line.strip() == statement)
    column = line.index(expression) + 1
    return start + index, column, start + index, column + len(expression)


def _span_position(span):
    return span.line, span.column, span.end_line, span.end_column


def test_source_and_ir_call_spans_use_one_based_columns(calls):
    # Generated expression calls must retain the original one-based start/end columns on the
    # emitted IR.
    M = calls

    @M.function
    def direct():
        M.evaluate(M.call_extern("int32", "direct"))

    expected = _position(
        test_source_and_ir_call_spans_use_one_based_columns,
        'M.evaluate(M.call_extern("int32", "direct"))',
        'M.call_extern("int32", "direct")',
    )
    span = direct.body[0][1].span
    assert _span_position(span) == expected
    assert span.source_name.name == __file__


def test_nested_helper_spans_preserve_caller_and_definition_columns(calls):
    # Nested macros must compose all caller and definition ranges instead of replacing or
    # duplicating them.
    M = calls

    @M.inline
    def inner():
        M.evaluate(M.call_extern("int32", "nested"))

    @M.inline
    def outer():
        inner()

    @M.function
    def main():
        outer()
        outer()

    test = test_nested_helper_spans_preserve_caller_and_definition_columns
    expected = [
        _position(test, "outer()", "outer()"),
        _position(test, "inner()", "inner()"),
        _position(
            test,
            'M.evaluate(M.call_extern("int32", "nested"))',
            'M.call_extern("int32", "nested")',
        ),
    ]
    calls = [value for _, value in main.body if isinstance(value, ir.Call)]
    assert len(calls) == 2
    for index, value in enumerate(calls):
        span = value.span
        assert isinstance(span, ir.SequentialSpan)
        caller = expected[0]
        actual_expected = [
            (caller[0] + index, caller[1], caller[2] + index, caller[3]),
            *expected[1:],
        ]
        assert [_span_position(item) for item in span.spans] == actual_expected
        assert all(item.source_name.name == __file__ for item in span.spans)


@pytest.fixture
def gallery(monkeypatch, spanned_language):
    monkeypatch.setitem(sys.modules, "__main__", ModuleType("__main__"))
    monkeypatch.setitem(globals(), "__name__", "__main__")
    M = spanned_language.M
    M.store = lambda value: ir.Call(
        ir.GlobalVar("store"), [ir.prim.IntImm("int32", value)], ret_ty="int32"
    )
    return M


def test_gallery_classes_retain_distinct_source_locations(gallery):
    # Repeated class names in a fileless gallery must recover their own source, not the first match.
    M = gallery
    width = 4

    @I.ir_module
    class Repeated:
        @M.function
        def main(A: M.Tensor((width,), "int32")):
            M.store(11)

    first = Repeated

    @I.ir_module
    class Repeated:
        @M.function
        def main(A: M.Tensor((width,), "int32")):
            M.store(22)

    second = Repeated
    source, start = inspect.getsourcelines(test_gallery_classes_retain_distinct_source_locations)
    for result, value in ((first, 11), (second, 22)):
        function = result["main"]
        assert function.params[0].args[0].args[0] == (4,)
        store = function.body[0][1]
        line = next(i for i, text in enumerate(source) if text.strip() == f"M.store({value})")
        assert store.args[0].value == value
        assert store.span.line == start + line
        assert store.span.source_name.name == __file__


def _line_of(function, statement):
    lines, first = inspect.getsourcelines(function)
    return first + next(index for index, line in enumerate(lines) if line.strip() == statement)


def test_callee_arguments_and_keywords_evaluate_once_with_caller_context(language):
    # Nested callees and keyword arguments must run once in written order and retain the call
    # location.
    M = language.M
    seen = []
    result_value = Value("returned")

    def callee():
        seen.append("callee")

        def function(a, *, b):
            seen.append((a, b))
            return result_value

        return function

    def operand(value):
        seen.append(value)
        return value

    @M.function
    def main():
        callee()(operand(1), b=operand(2))

    assert seen == ["callee", 1, 2, (1, 2)]
    assert main.body[0][1] is result_value
    assert result_value.span[-1][1] == _line_of(
        test_callee_arguments_and_keywords_evaluate_once_with_caller_context,
        "callee()(operand(1), b=operand(2))",
    )


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
