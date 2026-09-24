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
"""Control-flow syntax preserves construction order and native frame ownership."""

from __future__ import annotations

# Script-local bindings and failing decorated definitions are intentionally observable in IR.
# ruff: noqa: F841
import inspect

import pytest
from minilang import Value

from tvm.script import ir as I
from tvm.script import tirx as T


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


def test_lexical_range_binding_calls_the_custom_iterator_once(language):
    # A lexical range alias must call the captured iterator once instead of the builtin range.
    M = language.M
    calls = []

    def custom_range(extent):
        calls.append(extent)
        return M.grid(2)

    @M.function
    def main():
        range = custom_range
        for i in range(4):
            M.record(i)

    assert calls == [4]
    variable = main.body[0][1]
    assert variable.op == "loop" and variable.args == (2,) and variable.name == "i"


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
    # Invalid loop placement must be rejected, while disabled checks preserve the original IR.
    from tvm import ir, tirx

    @T.prim_func(check_well_formed=False)
    def invalid():
        T.evaluate(tirx.break_loop())

    # This is exactly the native node emitted by source `break`; constructing
    # the intrinsic explicitly keeps the surrounding Python definition valid.
    ir.assert_structural_equal(invalid.body, tirx.Evaluate(tirx.break_loop()))
    with pytest.raises(ValueError, match="requires an enclosing loop"):

        @T.prim_func
        def rejected():
            T.evaluate(tirx.break_loop())

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
