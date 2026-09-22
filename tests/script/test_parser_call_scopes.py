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
"""Uniform source-call evaluation and exception-safe caller locations."""

import pytest

from tvm import ir, tirx
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import parser_support as PS
from tvm.script import tirx as T


def loc(line, name="calls.py"):
    return (name, line, line, 1, 20)


def span_lines(value):
    spans = value.span.spans if isinstance(value.span, ir.SequentialSpan) else [value.span]
    return [span.line for span in spans]


def test_call_scope_keeps_result_identity_and_restores_after_exception():
    seen = []
    with IRBuilder():
        value = tirx.IntImm("int32", 1)
        assert PS.with_at_scope(loc(1), lambda: (seen.append(1), value)[1]) is value
        assert seen == [1]
        with pytest.raises(ValueError, match="failure"):
            PS.with_at_scope(loc(2), lambda: (_ for _ in ()).throw(ValueError("failure")))
        other = PS.at(loc(3), tirx.IntImm("int32", 2))
        assert span_lines(other) == [3]


def test_scoped_helper_multiple_emissions_and_normal_return():
    marker = object()
    with IRBuilder() as builder:
        with T.function():
            T.func_name("multiple")

            def helper():
                PS.with_at_scope(loc(8), lambda: T.evaluate(1))
                PS.with_at_scope(loc(9), lambda: T.evaluate(2))
                return marker

            assert PS.with_at_scope(loc(4), helper) is marker
    statements = builder.get().body.seq
    assert len(statements) == 2
    assert span_lines(statements[0]) == [4, 8]
    assert span_lines(statements[1]) == [4, 9]


def test_source_call_callee_arguments_and_keywords_evaluate_once_in_order():
    seen = []

    def callee():
        seen.append("callee")

        def function(a, *, b):
            seen.append((a, b))
            return a + b

        return function

    def operand(value):
        seen.append(value)
        return value

    source = """
@T.prim_func
def main():
    T.evaluate(callee()(operand(1), b=operand(2)))
"""
    result = parser.parse(source, extra_vars={"callee": callee, "operand": operand})
    assert seen == ["callee", 1, 2, (1, 2)]
    assert isinstance(result.body, tirx.Evaluate)
    assert result.body.value.value == 3


def test_module_source_calls_have_context_before_annotations():
    seen = []

    def record():
        node = tirx.IntImm("int32", 3)
        IRBuilder.current()._set_current_source_span(node)
        seen.append(node.span)
        return {}

    module = parser.parse(
        """
@I.ir_module
class Module:
    I.module_attrs(record())
    @T.prim_func
    def main():
        T.evaluate(1)
""",
        extra_vars={"record": record},
    )
    assert "main" in module
    assert len(seen) == 1
    assert seen[0] is not None
    assert seen[0].line == 4


def test_source_inline_keeps_caller_and_each_definition_location():
    function = parser.parse(
        """
@T.inline
def inner(x):
    T.evaluate(x)
    T.evaluate(x + 1)
@T.prim_func
def main():
    inner(1)
""",
        filename="inline_scope.py",
    )
    statements = function.body.seq
    assert len(statements) == 2
    assert span_lines(statements[0]) == [8, 4]
    assert span_lines(statements[1]) == [8, 5]
