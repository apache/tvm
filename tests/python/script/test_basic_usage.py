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
"""Shared parser basic usage."""

from __future__ import annotations

import inspect

import pytest

# Script-local bindings are observed through the constructed IR.
# ruff: noqa: F841
from minilang import Value

from tvm import DataType, error, ir
from tvm.ir import prim
from tvm.ir._overload_prim_expr import EqualOp
from tvm.script import ir as I
from tvm.script.parser import entry


def test_function(language):
    M = language.M

    @M.function
    def identity(x: M.Tensor((4,))) -> M.Tensor((4,)):
        return x

    assert identity.name == "identity"
    assert identity.params[0].args[0].args[0] == (4,)
    assert identity.ret_type.args[0] == (4,)
    assert identity.body == [("return", identity.params[0])]


def test_module(language):
    M = language.M

    # An empty source class must still produce a module without inventing functions.
    @I.ir_module
    class Empty:
        pass

    assert Empty == {}

    @I.ir_module
    class Module:
        @M.function
        def first(x: M.Tensor((4,))):
            return x

        @M.function
        def second(y: M.Tensor((4,))):
            return Module.first(y)

    assert set(Module) == {"first", "second"}
    assert Module["first"].body == [("return", Module["first"].params[0])]
    returned = Module["second"].body[0]
    assert returned[0] == "return"
    call = returned[1]
    assert call.op == "call" and call.args[0].args == ("first",)
    assert call.args[1] is Module["second"].params[0]
    assert Module["first"].params[0].args[0].args[0] == (4,)
    assert Module["second"].params[0].args[0].args[0] == (4,)


def test_unmarked_expressions_build_both_arms_in_source_order(language):
    # Unmarked selections build IR and evaluate both arms once in source order.
    M = language.M
    seen = []
    condition = Value("condition")

    def operand(value):
        seen.append(value)
        return value

    @M.function
    def main():
        operand(1) if condition else operand(2)
        condition and operand(3)
        condition or operand(4)

    assert seen == [1, 2, 3, 4]
    assert [value.op for _, value in main.body] == ["select", "and", "or"]
    assert all(value.args[0] is condition for _, value in main.body)


def test_ordinary_tuple_and_outer_branch_assignments_still_store(language):
    # Unpacking and branch writes must update existing mutable cells rather than replace them.
    M = language.M
    first, second = object(), object()
    calls = []

    def values():
        calls.append("values")
        return first, second

    @M.function
    def main():
        a = M.cell()
        b = M.cell()
        a, b = values()
        if M.value():
            a = first
        else:
            a = second
        x = 0
        x = 1
        x += 2
        b += 3
        M.record(x)

    declarations = dict(operands for kind, operands in main.body if kind == "declare")
    stores = [operands for kind, operands in main.body if kind == "set"]
    assert calls == ["values"]
    assert stores[:-1] == [
        (declarations["a"], first),
        (declarations["b"], second),
        (declarations["a"], first),
        (declarations["a"], second),
    ]
    assert main.body[-1] == ("emit", 3)
    target, addition = stores[-1]
    assert target is declarations["b"]
    assert addition.op == "add" and addition.args == (target, 3)


def test_name_collision(language):
    # Generated helpers must not steal identifiers already bound in source.
    M = language.M

    @M.function
    def main(_builder0: M.Tensor((4,))):
        _fn0 = 5
        _build0 = 6
        _X1 = 7
        M.record(_builder0)
        M.record(_fn0 + _build0 + _X1)

    assert main.body[0][1] is main.params[0]
    assert main.body[1][1] == 18


def test_loop_targets_preserve_names_and_local_rebinding(language):
    # Scalar and unpacked loop targets must retain names, extents and ordinary rebinding semantics.
    M = language.M

    @M.function
    def main():
        for i in M.grid(4):
            M.record(i)
            i = 5
            M.record(i)
        for iters in M.grid(4, 5):
            M.record(iters)
        for head, *tail in M.grid(2, 3, 4):
            M.record((head, *tail))

    scalar, rebound, packed, unpacked = [value for _, value in main.body]
    assert scalar.op == "loop" and scalar.args == (4,) and scalar.name == "i"
    assert rebound == 5
    assert [value.name for value in packed] == ["iters_0", "iters_1"]
    assert [value.args for value in packed] == [(4,), (5,)]
    assert [value.name for value in unpacked] == ["head", "tail_0", "tail_1"]
    assert [value.args for value in unpacked] == [(2,), (3,), (4,)]


def test_conditional_outputs_share_one_native_frame_result(language):
    # A value-producing conditional must return the native frame result without rebinding it.
    M = language.M
    M.__tvm_value_if__ = True
    condition, left, right = Value("condition"), Value("left"), Value("right")

    @M.function
    def main():
        if condition:
            y = left
        else:
            y = right
        M.record(y)

    output = main.body[0][1]
    assert output.op == "if" and output.args == (condition, left, right)


def test_void_branch_statements_need_no_synthetic_named_output(language):
    # Void branch effects must not require or introduce an artificial result binding.
    M = language.M
    M.__tvm_value_if__ = True
    condition = Value("condition")

    @M.function
    def main():
        if condition:
            M.record(None)
        else:
            M.record(None)
        M.record(9)

    assert [value for kind, value in main.body] == [None, None, 9]


def test_ordinary_iterator_binding_calls_the_custom_iterator_once(language):
    # An ordinary iterator alias calls its captured implementation exactly once.
    M = language.M
    calls = []

    def custom_range(extent):
        calls.append(extent)
        return M.grid(2)

    @M.function
    def main():
        iterator = custom_range
        for i in iterator(4):
            M.record(i)

    assert calls == [4]
    variable = main.body[0][1]
    assert variable.op == "loop" and variable.args == (2,) and variable.name == "i"


def test_body_annotation_reads_a_preceding_ordinary_local(language):
    # A body annotation must read the current local value, not a stale enclosing capture.
    M = language.M
    annotations = []

    def annotation(shape):
        annotations.append(shape)
        return M.Tensor(shape)

    M.annotation = annotation

    @M.function
    def main():
        shape = (4,)
        value: M.annotation(shape) = 1
        M.record(value)
        shape = (8,)
        M.record(shape)

    assert annotations == [(4,)]
    assert main.body == [("emit", 1), ("emit", (8,))]


def test_native_concise_scopes_unwind_with_their_parent():
    # Nested concise thread scopes must preserve the original variables in the constructed IR.
    from tvm import tirx

    variables = []

    def observe(*items):
        variables.extend(items)

    @T.prim_func
    def main():
        bx = T.launch_thread("blockIdx.x", 2)
        tx = T.launch_thread("threadIdx.x", 32)
        observe(bx, tx)
        T.evaluate(bx + tx)

    bx, tx = variables
    body = main.body
    assert isinstance(body, tirx.AttrStmt) and isinstance(body.body, tirx.AttrStmt)
    assert body.node.var.same_as(bx) and body.body.node.var.same_as(tx)
    assert body.body.body.value.a.same_as(bx) and body.body.body.value.b.same_as(tx)


def test_loop_control_validation_preserves_valid_and_unchecked_ir():
    # Invalid loop placement must be rejected, while direct IR construction preserves the node.
    from tvm import ir, tirx

    invalid = tirx.PrimFunc(params=[], body=tirx.Break())
    ir.assert_structural_equal(invalid.body, tirx.Break())
    assert not tirx.analysis.verify_well_formed(invalid, assert_mode=False)
    with pytest.raises(error.InternalError, match="requires an enclosing loop"):
        tirx.analysis.verify_well_formed(invalid)

    @T.prim_func
    def valid():
        for i in range(2):
            break

    assert isinstance(valid.body, tirx.For)
    ir.assert_structural_equal(valid.body.body, invalid.body)

    @I.ir_module(check_well_formed=False, extra_vars={"invalid": invalid})
    class Unchecked:
        bad = invalid

    assert Unchecked["bad"].same_as(invalid)
    with pytest.raises(ValueError, match="requires an enclosing loop"):

        @I.ir_module(extra_vars={"invalid": invalid})
        class Rejected:
            bad = invalid


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
    seen = []

    def consume(value):
        seen.append(value)
        return 0

    @M.function
    def main():
        consume(M.constexpr(left < right))

    assert len(seen) == 1 and seen[0] is result


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


def test_nested_unmarked_statement_retains_ir_frame(language):
    # A constexpr outer branch must not turn its unmarked inner branch into Python control flow.
    M = language.M

    @M.function
    def main(condition: M.value):
        if I.constexpr(True):
            if condition:
                M.record(1)
            else:
                M.record(2)

    assert main.body == [("emit", 1), ("emit", 2)]


def test_ir_optional_binding_survives_skipped_host_assignment(language):
    # A skipped host assignment must not corrupt an optional binding from an IR branch.
    M = language.M

    @M.function
    def main(condition: M.value):
        if condition:
            x = M.value(1)
        if I.constexpr(False):
            x = M.value(2)
        x = M.value(3)
        M.record(x)

    assert main.body[-1][1].args == (3,)


def test_source_logical_operands_skip_at_construction(language):
    # Boolean operators inside constexpr must skip the unused source operands.
    M = language.M

    def fail():
        raise AssertionError("skipped source operand was evaluated")

    @M.function
    def main():
        M.record(1 if I.constexpr(False and fail()) else 2)
        M.record(3 if I.constexpr(True or fail()) else 4)

    assert [value for _, value in main.body] == [2, 3]


def test_unmarked_logical_chain_preserves_left_association(language):
    # Unmarked logical chains must retain their left-associated native expression structure.
    M = language.M

    @M.function
    def main(a: M.value(), b: M.value(), c: M.value()):
        M.record(a and b and c)

    expression = main.body[0][1]
    assert expression.op == "and"
    assert expression.args[1] is main.params[2]
    assert expression.args[0].op == "and"
    assert expression.args[0].args == tuple(main.params[:2])


def test_ir_branch_incoming_read_and_rebinding_obeys_python_scope(language):
    # A branch-local write must not read the outer binding before its own initialization.
    M = language.M
    with pytest.raises(NameError, match="y"):

        @M.function
        def main(condition: M.value(), x: M.value()):
            y = x
            if condition:
                y = y + x
            else:
                y = x
            return y


def test_datatype_value_roundtrip(language):
    # A DataType value needs only shared IR: the minimal language parses its shared spelling
    # into a root-typed constant, and the shared printer reproduces that spelling.
    spelling = 'I.dtype("float32")'
    source = f"@M.function\ndef main():\n    M.record({spelling})\n"
    value = entry.parse(source, extra_vars={"M": language.M, "I": I}).body[0][1]
    ir.assert_structural_equal(value, ir.DataTypeImm(DataType("float32")))
    printed = ir.IRModule(attrs={"dtype": value}).script()
    assert f'I.module_attrs({{"dtype": {spelling}}})' in printed


def test_mutating_stores_and_loop_control_keep_effect_order(language):
    # Stores evaluate RHS before target/index; augmented stores load the target before their RHS.
    M = language.M
    effects = []

    class Storage:
        def __init__(self):
            self.value = 0
            self.items = [0]

        def __getitem__(self, key):
            effects.append("read")
            return self.items[key]

        def __setitem__(self, key, value):
            self.items[key] = value

    storage = Storage()

    def receiver():
        effects.append("receiver")
        return storage

    def key():
        effects.append("key")
        return 0

    def increment(value=3):
        effects.append("increment")
        return value

    @M.function
    def main(condition: M.value()):
        first, second = (2, 4)
        receiver().value = first
        receiver()[key()] = increment(second)
        receiver().value += increment()
        receiver()[key()] += increment()
        while condition:
            assert not condition, "keep the assertion in the loop"
            for i in range(2):
                if condition:
                    continue
                else:
                    break
            break
        return

    assert storage.value == 5 and storage.items == [7]
    assert effects == [
        "receiver",
        "increment",
        "receiver",
        "key",
        "receiver",
        "increment",
        "receiver",
        "key",
        "read",
        "increment",
    ]
    assert [kind for kind, _ in main.body] == [
        "setattr",
        "setitem",
        "setattr",
        "setitem",
        "assert",
        "continue",
        "break",
        "break",
        "return",
    ]
    assertion, message = main.body[4][1]
    assert assertion.op == "not" and assertion.args[0] is main.params[0]
    assert message == "keep the assertion in the loop"
    assert main.body[-1] == ("return", None)


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


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("1, 3.14, True, 'str'", (1, 3.14, True, "str")),
        ("1 + 2, 1 - 2, 1 * 2, 1 / 2", (3, -1, 2, 0.5)),
        ("a + b, a - b, a * b, a / b", (3, -1, 2, 0.5)),
        ("func(a, b)", (3, -1, 2, 0.5)),
        (
            "values, values[1:], values[:5], values[1:5], values[1:5:2]",
            ([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [2, 3, 4, 5], [2, 4]),
        ),
    ],
    ids=["literal-types", "arithmetic", "captures", "callable", "slice-strides"],
)
def test_host_expression_values(language, expression, expected):
    calls = []

    def func(a, b):
        calls.append((a, b))
        return a + b, a - b, a * b, a / b

    source = "@M.function\ndef main():\n    M.record((" + expression + "))\n"
    function = entry.parse(
        source,
        extra_vars={
            "M": language.M,
            "a": 1,
            "b": 2,
            "func": func,
            "values": [1, 2, 3, 4, 5, 6],
        },
    )
    assert len(function.body) == 1
    kind, actual = function.body[0]
    assert kind == "emit" and isinstance(actual, tuple)
    assert actual == expected
    assert [type(value) for value in actual] == [type(value) for value in expected]
    assert calls == ([(1, 2)] if expression == "func(a, b)" else [])
