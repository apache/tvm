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
"""Ordinary mini-language functions and modules evaluate expressions and control flow.

Bindings, comparisons, loops and branches preserve values, identity and effect order.
"""

from __future__ import annotations

# Script-local bindings and failing decorated definitions are intentionally observable in IR.
# ruff: noqa: F841
import ast
import inspect

import pytest
from minilang import Value

from tvm import ir
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


def test_missing_parameter_annotation_keeps_source_range(language):
    # An unannotated script parameter must report that parameter before constructing IR.
    M = language.M
    with pytest.raises(SyntaxError, match="requires an annotation") as caught:

        @M.function
        def main(value):
            pass

    error = caught.value
    # The outer test's fixture argument is not the offending script argument.
    lines, first = inspect.getsourcelines(test_missing_parameter_annotation_keeps_source_range)
    node = next(
        n
        for n in ast.walk(ast.parse("".join(lines)))
        if isinstance(n, ast.arg) and n.arg == "value"
    )
    expected = (
        __file__,
        first + node.lineno - 1,
        node.col_offset + 1,
        first + node.end_lineno - 1,
        node.end_col_offset + 1,
    )
    assert type(error) is SyntaxError
    assert (
        error.filename,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    ) == expected
    assert not language.functions

    # The string entry must report the same missing annotation in its original source.
    source = "@M.function\ndef main(value) -> None:\n    M.record(0)\n"
    with pytest.raises(SyntaxError, match="requires an annotation") as caught:
        entry.parse(source, extra_vars={"M": M})

    error = caught.value
    assert type(error) is SyntaxError
    assert (
        error.filename,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    ) == ("<str>", 2, 10, 2, 15)
    assert not language.functions


def test_invalid_quoted_annotation_keeps_source_range(language):
    # Even malformed quoted annotations report the original literal without decoding it.
    M = language.M
    with pytest.raises(SyntaxError, match="Quoted annotations are not supported") as caught:

        @M.function
        def main(value: "invalid +"):  # noqa: F722
            pass

    error = caught.value
    lines, first = inspect.getsourcelines(test_invalid_quoted_annotation_keeps_source_range)
    node = next(
        n
        for n in ast.walk(ast.parse("".join(lines)))
        if isinstance(n, ast.Constant) and n.value == "invalid +"
    )
    expected = (
        __file__,
        first + node.lineno - 1,
        node.col_offset + 1,
        first + node.end_lineno - 1,
        node.end_col_offset + 1,
    )
    assert type(error) is SyntaxError
    assert (
        error.filename,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    ) == expected
    assert not language.functions


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


def test_policies_leave_computed_arguments_and_shorthand_strings_alone(language):
    # Computed arguments must run once without treating their returned strings as symbolic syntax.
    M = language.M
    seen = []
    concrete = object()

    def value(label, result):
        seen.append(label)
        return result

    @M.function
    def main():
        M.Tensor(
            value("shape", (4,)), dtype=value("dtype", "float32"), device=value("device", concrete)
        )
        M.Tensor("float32", placement="literal[0]")

    assert seen == ["shape", "dtype", "device"]
    assert main.body[0][1].args == ((4,), "float32", concrete, "S[0]")
    assert main.body[1][1].args == ("float32", "float32", None, "literal[0]")


def test_nested_policy_and_starred_calls_are_evaluated_once(language):
    # Nested marked calls and starred arguments must preserve their values and evaluation count.
    M = language.M
    seen = []
    mesh = object()
    language.global_infos["mesh[0]"] = mesh
    values = (1, 2)
    n = M.dynamic("n")

    def outer(value):
        seen.append(value)
        return value

    def collect(*values, other):
        return values, other

    @M.function
    def main():
        outer(M.Tensor((n,), device="mesh[0]"))
        collect(*values, other=3)

    assert len(seen) == 1 and seen[0].args[2] is mesh
    dimension = seen[0].args[0][0]
    assert dimension.op == "symbol" and dimension.name == "n"
    assert main.body[1][1] == ((1, 2), 3)


def test_constexpr_is_lazy_and_executes_in_parent_scope(language):
    # Compile-time branches keep Python scope and short-circuit without visiting unselected
    # operands.
    M = language.M
    seen = []

    def choose():
        seen.append("condition")
        return True

    def operand(value):
        seen.append(value)
        return value

    def invalid():
        pytest.fail("an unselected constexpr arm ran")

    @M.function
    def main():
        if M.constexpr(choose()):
            x = 7
        else:
            invalid()
        M.record(x)
        M.record(operand(1) if I.constexpr(operand(True)) else invalid())
        M.record(I.constexpr(operand(0)) and invalid())
        M.record(I.constexpr(operand(4)) or invalid())
        M.record(I.constexpr(operand(2)) and operand(7))
        M.record(I.constexpr(operand(0)) or operand(8))

    assert seen == ["condition", True, 1, 0, 4, 2, 7, 0, 8]
    assert [value for kind, value in main.body] == [7, 1, 0, 4, 7, 8]


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


def test_ordinary_callable_aliases_update_mutable_targets(language):
    # A local callable must shadow its ambient namesake and preserve ordinary mutable stores.
    M = language.M
    marker, calls = object(), []

    def axis_alias():
        pytest.fail("the shadowed ambient callable ran")

    def ordinary():
        calls.append("ordinary")
        return marker

    @M.function
    def main():
        axis_alias = ordinary
        cell = M.cell()
        cell = axis_alias()
        M.record(cell)

    declaration = next(operands[1] for kind, operands in main.body if kind == "declare")
    stores = [operands for kind, operands in main.body if kind == "set"]
    assert calls == ["ordinary"]
    assert len(stores) == 1 and stores[0][0] is declaration and stores[0][1] is marker
    assert main.body[-1] == ("emit", declaration)


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


def test_bare_callable_alias_does_not_acquire_constexpr_syntax(language):
    # A bare callable alias must not silently acquire constexpr syntax from the original marker.
    M = language.M
    marker = I.constexpr
    with pytest.raises(TypeError, match="syntax marker"):

        @M.function
        def main():
            if marker(True):
                M.record(1)


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


def test_conditional_branches_require_matching_output_names(language):
    # Value-producing branches must agree on their output name and locate the mismatched assignment.
    M = language.M
    M.__tvm_value_if__ = True
    condition, left, right = Value("condition"), Value("left"), Value("right")
    with pytest.raises(SyntaxError, match="same named output") as caught:

        @M.function
        def main():
            if condition:
                y = left
            else:
                z = right

    error = caught.value
    lines, first = inspect.getsourcelines(test_conditional_branches_require_matching_output_names)
    line = first + next(i for i, text in enumerate(lines) if text.strip() == "z = right")
    assert (error.filename, error.lineno, error.end_lineno) == (__file__, line, line)
    assert (error.offset, error.end_offset) == (17, 26)
    assert not language.functions


def test_conditional_binding_can_be_assigned_after_skipped_branch(language):
    # A skipped constexpr assignment must not prevent a later ordinary assignment.
    M = language.M

    @M.function
    def main():
        if I.constexpr(False):
            x = 1
        x = 2
        M.record(x)

    assert main.body[-1] == ("emit", 2)


def test_constexpr_keeps_named_expression_unsupported(language):
    # Constexpr must reject a multiline assignment expression at its full original range.
    M = language.M
    with pytest.raises(SyntaxError, match="Unsupported expression: NamedExpr") as caught:

        @M.function
        def main():
            if I.constexpr(
                bool(
                    value := 1  # Keep the diagnostic range across two source lines.
                    + 2
                )
            ):
                M.record(value)

    error = caught.value
    lines, first = inspect.getsourcelines(test_constexpr_keeps_named_expression_unsupported)
    node = next(n for n in ast.walk(ast.parse("".join(lines))) if isinstance(n, ast.NamedExpr))
    assert type(error) is SyntaxError
    assert (
        error.filename,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    ) == (
        __file__,
        first + node.lineno - 1,
        node.col_offset + 1,
        first + node.end_lineno - 1,
        node.end_col_offset + 1,
    )
    assert not language.functions


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


def test_missing_host_binding_cannot_be_truth_tested(language):
    # Reading an unexecuted constexpr binding must raise before truth testing it.
    M = language.M
    with pytest.raises(NameError):

        @M.function
        def main():
            if I.constexpr(False):
                x = 1
            M.record(1 if I.constexpr(x) else 0)


def test_missing_host_if_binding_raises_before_branch_assignments(language):
    # An if-condition must reject its missing incoming value before either arm assigns that name.
    M = language.M
    with pytest.raises(NameError):

        @M.function
        def main():
            if I.constexpr(False):
                x = 1
            if I.constexpr(x):
                x = 2
            else:
                x = 3


def test_host_lambda_local_does_not_capture_optional_binding(language):
    # A lambda parameter must shadow an outer optional binding during host selection.
    M = language.M

    @M.function
    def main():
        if I.constexpr(False):
            x = 1
        if I.constexpr((lambda x: x)(True)):
            M.record(222)

    assert main.body[-1] == ("emit", 222)


def test_unselected_namespace_call_does_not_require_an_attribute(language):
    # An unselected source call must remain unevaluated even when its captured name resembles a
    # parser helper.
    M = language.M
    lanes = 1

    @M.function
    def main():
        M.record(M.ramp(0, 1, lanes) if I.constexpr(lanes > 1) else 0)

    assert main.body == [("emit", 0)]


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


def test_module_string_constants_keep_common_constructor_values():
    # Literal strings passed to common IR constructors must remain concrete values
    # when a real module definition is parsed.
    from tvm import ir
    from tvm.script import ir as I

    @I.ir_module
    class Module:
        I.module_attrs({"tag": I.StringImm("label"), "type": I.StringType()})

    expected = ir.IRModule(attrs={"tag": ir.StringImm("label"), "type": ir.StringType()})
    ir.assert_structural_equal(expected, Module)
    assert Module.attrs["tag"].value == "label"
    assert isinstance(Module.attrs["type"], ir.StringType)


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
