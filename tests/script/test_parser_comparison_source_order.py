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
"""Comparisons preserve source orientation while explicit constexpr retains host overloads."""

import pytest

from tvm import ir, tirx
from tvm.script import parser

OPERATORS = [("<", "LT"), ("<=", "LE"), (">", "GT"), (">=", "GE"), ("==", "EQ"), ("!=", "NE")]


@pytest.mark.parametrize("operator,kind", OPERATORS)
@pytest.mark.parametrize("literal_left", [True, False])
@pytest.mark.parametrize("track_span", [True, False])
def test_written_comparison_order(operator, kind, literal_left, track_span):
    expression = f"0 {operator} x" if literal_left else f"x {operator} 0"
    actual = parser.parse(
        f"@T.prim_func\ndef main(x: T.int32):\n    T.evaluate({expression})\n",
        track_span=track_span,
    )
    x = actual.params[0]
    operands = (0, x) if literal_left else (x, 0)
    ir.assert_structural_equal(actual.body.value, getattr(tirx, kind)(*operands))
    if track_span:
        assert actual.body.value.span is not None


@pytest.mark.parametrize("dtype", ["int64", "uint32", "float32", "int32x4", "float32x4"])
def test_literal_uses_ir_operand_type_and_lanes(dtype):
    actual = parser.parse(f"@T.prim_func\ndef main(x: T.{dtype}):\n    T.evaluate(0 < x)\n")
    value = actual.body.value
    assert isinstance(value, tirx.LT)
    assert str(value.a.ty) == str(value.b.ty) == dtype
    assert value.b.same_as(actual.params[0])


@pytest.mark.parametrize("operator,kind", OPERATORS)
@pytest.mark.parametrize("other_kind", ["host", "primitive", "iterator"])
def test_custom_host_comparison_result_is_preserved(operator, kind, other_kind):
    class Result:
        def __bool__(self):
            raise AssertionError("custom comparison result must not be truth-tested")

        def asobject(self):
            raise AssertionError("custom comparison result must not be converted")

    result = Result()
    calls = []

    class Host:
        pass

    def compare(self, other):
        calls.append(other)
        return result

    setattr(
        Host,
        {
            "LT": "__lt__",
            "LE": "__le__",
            "GT": "__gt__",
            "GE": "__ge__",
            "EQ": "__eq__",
            "NE": "__ne__",
        }[kind],
        compare,
    )

    def consume(value):
        assert value is result
        return 0

    other = "x" if other_kind == "primitive" else "right"
    right = (
        tirx.IterVar(None, "axis", tirx.IterVar.DataPar) if other_kind == "iterator" else object()
    )
    actual = parser.parse(
        "@T.prim_func\ndef main(x: T.int32):\n"
        f"    T.evaluate(consume(T.constexpr(left {operator} {other})))\n",
        extra_vars={"left": Host(), "right": right, "consume": consume},
    )
    assert len(calls) == 1
    if other_kind == "primitive":
        assert calls[0].same_as(actual.params[0])
    else:
        assert calls[0] is right


def test_numeric_subclass_keeps_custom_comparison():
    seen = []
    result = object()

    class HostInt(int):
        def __lt__(self, other):
            seen.append(other)
            return result

    def consume(value):
        assert value is result
        return 0

    parser.parse(
        "@T.prim_func\ndef main(x: T.int32):\n    T.evaluate(consume(T.constexpr(left < x)))\n",
        extra_vars={"left": HostInt(0), "consume": consume},
    )
    assert len(seen) == 1


def test_host_ordering_does_not_materialize_an_equality_result():
    result = tirx.Var("host_result", "int32") == 0

    class Host:
        def __lt__(self, other):
            return result

    def consume(value):
        assert value is result
        return 0

    parser.parse(
        "@T.prim_func\ndef main():\n    T.evaluate(consume(T.constexpr(left < right)))\n",
        extra_vars={"left": Host(), "right": Host(), "consume": consume},
    )


def test_chain_evaluates_source_operands_once_in_order():
    seen = []

    def operand(index, value):
        seen.append(index)
        return value

    actual = parser.parse(
        """
@T.prim_func
def main(x: T.int32, y: T.int32):
    T.evaluate(operand(0, 0) < operand(1, x) <= operand(2, y) != operand(3, 3))
""",
        extra_vars={"operand": operand},
    )
    x, y = actual.params
    expected = tirx.And(tirx.LT(0, x), tirx.And(tirx.LE(x, y), tirx.NE(y, 3)))
    ir.assert_structural_equal(actual.body.value, expected)
    assert seen == [0, 1, 2, 3]


def test_chain_binds_effectful_middle_operand_once():
    actual = parser.parse("""
@T.prim_func
def main():
    T.evaluate(0 < T.call_extern("int32", "middle") < 10)
""")
    value = actual.body.value
    assert isinstance(value, tirx.Let)
    assert isinstance(value.value, ir.Call)
    expected = tirx.And(tirx.LT(0, value.var), tirx.LT(value.var, 10))
    ir.assert_structural_equal(value.body, expected)


def test_constexpr_keeps_python_comparison_and_chain_short_circuit():
    seen = []

    def operand(index, value):
        seen.append(index)
        return value

    def invalid():
        raise AssertionError("constexpr comparison chain must short-circuit")

    actual = parser.parse(
        """
@T.prim_func
def main(x: T.int32):
    if T.constexpr(operand(0, 2) < operand(1, 1) < invalid()):
        T.evaluate(0)
    elif T.constexpr(x == x):
        T.evaluate(1)
""",
        extra_vars={"operand": operand, "invalid": invalid},
    )
    assert seen == [0, 1]
    assert actual.body.value.value == 1


@pytest.mark.parametrize("operator,kind", OPERATORS)
@pytest.mark.parametrize("dtype", ["int64", "float32", "int64x4"])
def test_comparisons_use_native_promotion_and_broadcast(operator, kind, dtype):
    actual = parser.parse(
        f"@T.prim_func\ndef main(x: T.int32, y: T.{dtype}):\n    T.evaluate(x {operator} y)\n"
    )
    x, y = actual.params
    lhs = tirx.Broadcast(tirx.Cast("int64", x), 4) if dtype.endswith("x4") else tirx.Cast(dtype, x)
    expected = getattr(tirx, kind)(lhs, y)
    ir.assert_structural_equal(actual.body.value, expected)


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_iterator_comparison_keeps_written_operand_order(operator, kind):
    calls = []

    def axis(value):
        calls.append(value)
        return tirx.IterVar(None, value, tirx.IterVar.DataPar)

    actual = parser.parse(
        f"@T.prim_func\ndef main(x: T.int32):\n    T.evaluate(0 {operator} axis(x))\n",
        extra_vars={"axis": axis},
    )
    assert len(calls) == 1
    assert calls[0].same_as(actual.params[0])
    ir.assert_structural_equal(actual.body.value, getattr(tirx, kind)(0, actual.params[0]))


@pytest.mark.parametrize("operator,kind", OPERATORS)
@pytest.mark.parametrize("dialect", ["T", "R"])
def test_unmarked_host_comparison_never_calls_python_overload(operator, kind, dialect):
    from tvm.error import DiagnosticError

    calls = []

    class Host:
        pass

    def comparison(self, other):
        calls.append(other)
        return True

    setattr(
        Host,
        {
            "LT": "__lt__",
            "LE": "__le__",
            "GT": "__gt__",
            "GE": "__ge__",
            "EQ": "__eq__",
            "NE": "__ne__",
        }[kind],
        comparison,
    )
    decorator = "T.prim_func" if dialect == "T" else "R.function"
    statement = (
        "T.evaluate(left " + operator + " right)"
        if dialect == "T"
        else "return left " + operator + " right"
    )
    with pytest.raises(DiagnosticError, match="(PrimExpr|primitive|convert|type)"):
        parser.parse(
            f"@{decorator}\ndef main():\n    {statement}\n",
            extra_vars={"left": Host(), "right": Host()},
        )
    assert calls == []


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_relax_tensor_comparison_constructs_written_operator(operator, kind):
    from tvm import relax

    operation = {
        "LT": "less",
        "LE": "less_equal",
        "GT": "greater",
        "GE": "greater_equal",
        "EQ": "equal",
        "NE": "not_equal",
    }[kind]
    actual = parser.parse(
        '@R.function\ndef main(x: R.Tensor((2,), "float32"), y: R.Tensor((2,), "float32")):\n'
        f"    return x {operator} y\n",
        track_span=True,
    )
    call = actual.body.blocks[0].bindings[0].value
    expected = relax.BlockBuilder().normalize(getattr(relax.op, operation)(*actual.params))
    ir.assert_structural_equal(call, expected)
    assert call.span is not None


@pytest.mark.parametrize("operator,kind", OPERATORS)
def test_relax_primitive_comparison_is_concrete_ir(operator, kind):
    actual = parser.parse(
        f'@R.function\ndef main(x: R.Prim("int32")):\n    return 0 {operator} x\n'
    )
    ir.assert_structural_equal(actual.body.body, getattr(tirx, kind)(0, actual.params[0]))
