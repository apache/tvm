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
"""Syntax policies preserve explicit host evaluation and namespace behavior."""

from __future__ import annotations

# Script-local bindings and failing decorated definitions are intentionally observable in IR.
import ast
import inspect

import pytest

from tvm.script import ir as I
from tvm.script.parser import protocol_registry as registry


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

    def outer(value):
        seen.append(value)
        return value

    def collect(*values, other):
        return values, other

    @M.function
    def main():
        outer(M.Tensor(("n",), device="mesh[0]"))
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


def test_bare_callable_alias_does_not_acquire_constexpr_syntax(language):
    # A bare callable alias must not silently acquire constexpr syntax from the original marker.
    M = language.M
    marker = registry.constexpr
    with pytest.raises(TypeError, match="syntax marker"):

        @M.function
        def main():
            if marker(True):
                M.record(1)


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


def test_unselected_namespace_call_does_not_require_an_attribute(language):
    # An unselected source call must remain unevaluated even when its captured name resembles a
    # parser helper.
    M = language.M
    lanes = 1

    @M.function
    def main():
        M.record(M.ramp(0, 1, lanes) if I.constexpr(lanes > 1) else 0)

    assert main.body == [("emit", 0)]


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
