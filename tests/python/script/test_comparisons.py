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
"""Comparison syntax preserves written operations, types, and operand identity."""

from __future__ import annotations

# Script-local bindings and failing decorated definitions are intentionally observable in IR.
import inspect

import pytest

from tvm import ir
from tvm.ir import prim
from tvm.ir._overload_prim_expr import EqualOp


def test_written_comparison_order(primitive_language):
    # A leading host literal must not reverse the written native comparison or its source span.
    M = primitive_language.M
    x = ir.Var("x", "int32")

    @M.function
    def main():
        0 < x

    actual = main.body[0][1]
    ir.assert_structural_equal(actual, prim.LT(0, x))
    lines, start = inspect.getsourcelines(test_written_comparison_order)
    line = start + next(i for i, text in enumerate(lines) if text.strip() == "0 < x")
    assert (actual.span.line, actual.span.end_line, actual.span.column, actual.span.end_column) == (
        line,
        line,
        9,
        14,
    )


def test_literal_uses_ir_operand_type_and_lanes(primitive_language):
    # Scalar literals must adopt the vector IR operand type and lanes.
    M = primitive_language.M
    x = ir.Var("x", "float32x4")

    @M.function
    def main():
        0 < x

    value = main.body[0][1]
    assert isinstance(value, prim.LT)
    assert str(value.a.ty) == str(value.b.ty) == "float32x4"
    assert value.b.same_as(x)


def test_custom_host_comparison_result_is_preserved(primitive_language):
    # Compile-time comparison must preserve a custom host result instead of forcing IR conversion.
    M = primitive_language.M
    result = object()
    calls = []

    class HostInt(int):
        def __lt__(self, other):
            calls.append(other)
            return result

    left, right = HostInt(0), ir.Var("x", "int32")

    def consume(value):
        assert value is result
        return 0

    @M.function
    def main():
        consume(M.constexpr(left < right))

    assert len(calls) == 1 and calls[0].same_as(right)


def test_host_ordering_does_not_materialize_an_equality_result(primitive_language):
    # An ordering method may return an equality proxy; constexpr must preserve that object.
    M = primitive_language.M
    result = ir.Var("host_result", "int32") == 0

    class Host:
        def __lt__(self, other):
            return result

    left, right = Host(), Host()

    def consume(value):
        assert value is result
        return 0

    @M.function
    def main():
        consume(M.constexpr(left < right))


def test_simple_chain_and_complex_operand_rejection(primitive_language):
    # Simple comparison chains preserve order; unsupported effectful operands fail before execution.
    M = primitive_language.M
    x, y = ir.Var("x", "int32"), ir.Var("y", "int32")

    @M.function
    def main():
        -1 < x <= y != +3

    expected = prim.And(prim.LT(-1, x), prim.And(prim.LE(x, y), prim.NE(y, 3)))
    ir.assert_structural_equal(main.body[0][1], expected)

    def operand():
        pytest.fail("unsupported chain evaluated its operand")

    with pytest.raises(SyntaxError, match="chain") as caught:

        @M.function
        def invalid():
            operand() < x < y

    lines, start = inspect.getsourcelines(test_simple_chain_and_complex_operand_rejection)
    line = start + next(i for i, text in enumerate(lines) if text.strip() == "operand() < x < y")
    assert (caught.value.filename, caught.value.lineno, caught.value.offset) == (__file__, line, 13)


def test_constexpr_keeps_python_comparison_and_chain_short_circuit(primitive_language):
    # Compile-time comparison chains must short-circuit and retain native identity equality
    # semantics.
    M = primitive_language.M
    seen = []
    x = ir.Var("x", "int32")

    def operand(index, value):
        seen.append(index)
        return value

    def invalid():
        raise AssertionError("constexpr comparison chain must short-circuit")

    @M.function
    def main():
        if M.constexpr(operand(0, 2) < operand(1, 1) < invalid()):
            0
        elif M.constexpr(x == x):
            1

    assert seen == [0, 1]
    assert main.body == [("emit", 1)]


def test_comparisons_use_native_promotion_and_broadcast(primitive_language):
    # Mixed scalar/vector comparisons must use native promotion and broadcast.
    M = primitive_language.M
    x, y = ir.Var("x", "int32"), ir.Var("y", "int64x4")

    @M.function
    def main():
        x < y

    expected = prim.LT(prim.Broadcast(prim.Cast("int64", x), 4), y)
    ir.assert_structural_equal(main.body[0][1], expected)


def test_symbolic_equality_reaches_typed_consumer_once_in_order(primitive_language):
    # Symbolic equality must reach its typed consumer once, after operands run in written order.
    M = primitive_language.M
    values = [ir.Var("x", "int32"), ir.Var("y", "int32")]
    seen, comparisons = [], []

    def operand(index):
        seen.append(index)
        return values[index]

    def consume(value):
        assert isinstance(value, prim.EQ)
        assert str(value.ty) == "bool"
        comparisons.append(value)
        return value

    @M.function
    def main():
        consume(operand(0) == operand(1))

    assert seen == [0, 1]
    assert len(comparisons) == 1
    assert comparisons[0].a.same_as(values[0]) and comparisons[0].b.same_as(values[1])
    assert comparisons[0].span is not None


def test_python_comparisons_remain_python_booleans(primitive_language):
    # Compile-time comparisons of ordinary containers and strings must return actual Python
    # booleans.
    M = primitive_language.M
    seen = []

    def consume(value):
        assert type(value) is bool
        seen.append(value)
        return int(value)

    @M.function
    def main():
        consume(M.constexpr([1, 2] == [1, 2]))
        consume(M.constexpr("x" != "y"))

    assert seen == [True, True]


def test_host_equality_result_is_not_arbitrarily_converted(primitive_language):
    # A custom equality proxy must not be materialized merely because its base resembles IR
    # equality.
    M = primitive_language.M

    class CustomEquality(EqualOp):
        def asobject(self):
            raise AssertionError("custom equality subclasses are host values")

    result = CustomEquality(ir.Var("x", "int32"), 0)

    class Host:
        def __eq__(self, other):
            return result

    left, right = Host(), Host()
    seen = []

    def consume(value):
        assert value is result
        seen.append(value)
        return 0

    @M.function
    def main():
        consume(M.constexpr(left == right))

    assert len(seen) == 1 and seen[0] is result


def test_numeric_relations_keep_each_written_operation(primitive_language):
    # A chained bound and its reversed bounds must retain all six written relations.
    M = primitive_language.M
    lower, value, upper = [ir.Var(name, "int32") for name in ("lower", "value", "upper")]

    @M.function
    def main():
        lower < value <= upper
        upper > value >= lower
        value == lower
        value != upper

    expected = [
        prim.And(prim.LT(lower, value), prim.LE(value, upper)),
        prim.And(prim.GT(upper, value), prim.GE(value, lower)),
        prim.EQ(value, lower),
        prim.NE(value, upper),
    ]
    for (_, actual), reference in zip(main.body, expected):
        ir.assert_structural_equal(actual, reference)
    assert len(main.body) == len(expected)
