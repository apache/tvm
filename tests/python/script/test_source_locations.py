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
import inspect
import sys
from types import ModuleType

import pytest
from minilang import Value

from tvm import ir
from tvm.ir import prim
from tvm.script import ir as I


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
