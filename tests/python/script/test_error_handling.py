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
"""Shared parser error handling."""

from __future__ import annotations

import ast
import inspect
import traceback

import pytest

# Script-local bindings are observed through the constructed IR.
# ruff: noqa: F841
from minilang import Value

from tvm import ir
from tvm.ir import prim
from tvm.script import ir as I
from tvm.script.parser import entry


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


def test_missing_host_binding_raises_before_use(language):
    # Reading an unexecuted constexpr binding must raise before truth testing it, including an
    # if-condition whose arms would assign that name.
    M = language.M
    with pytest.raises(NameError, match="'x'"):

        @M.function
        def expression():
            if I.constexpr(False):
                x = 1
            M.record(1 if I.constexpr(x) else 0)

    with pytest.raises(NameError, match="'x'"):

        @M.function
        def statement():
            if I.constexpr(False):
                x = 1
            if I.constexpr(x):
                x = 2
            else:
                x = 3


def test_nested_function_failure_preserves_error_and_recovers(spanned_language):
    # A nested failure keeps the original error/call sites and leaves the next script clean.
    language = spanned_language
    M = language.M
    seen = []
    failure = ValueError("inner body failure")

    def observe(label, outer, current, scope):
        seen.append((label, outer, current, scope))
        if label == "inner":
            raise failure

    with pytest.raises(ValueError) as caught:

        @M.function
        def main(x: M.Tensor((4,))):
            scope = 1
            observe("outer_before", x, x, scope)

            @M.function
            def inner(y: M.Tensor((4,))):
                scope = 2
                observe("inner", x, y, scope)

            observe("outer_after", x, x, scope)

    assert caught.value is failure and type(caught.value) is ValueError
    lines, start = inspect.getsourcelines(test_nested_function_failure_preserves_error_and_recovers)
    statement = 'observe("inner", x, y, scope)'
    index, line = next((i, line) for i, line in enumerate(lines) if line.strip() == statement)
    call_line, column = start + index, line.index(statement)
    frames = traceback.extract_tb(caught.value.__traceback__)
    calls = [frame for frame in frames if frame.filename == __file__ and frame.lineno == call_line]
    assert calls
    helper_frames = [frame for frame in frames if frame.name == "observe"]
    assert len(helper_frames) == 1 and helper_frames[0].filename == __file__
    raise_index, raise_source = next(
        (i, line) for i, line in enumerate(lines) if line.strip() == "raise failure"
    )
    assert helper_frames[0].lineno == start + raise_index
    if getattr(calls[-1], "colno", None) is not None:
        assert (calls[-1].colno, calls[-1].end_lineno, calls[-1].end_colno) == (
            column,
            call_line,
            column + len(statement),
        )
        raise_column = raise_source.index("raise failure")
        assert (
            helper_frames[0].colno,
            helper_frames[0].end_lineno,
            helper_frames[0].end_colno,
        ) == (raise_column, start + raise_index, raise_column + len("raise failure"))

    outer, inner = language.functions["main"], language.functions["inner"]
    assert len(outer.params) == len(inner.params) == 1
    assert seen == [
        ("outer_before", outer.params[0], outer.params[0], 1),
        ("inner", outer.params[0], inner.params[0], 2),
    ]
    M.node = lambda value: prim.IntImm("int32", value)

    @M.function
    def recovered(value: M.Tensor((2,))):
        M.record(M.node(23))
        return value

    assert recovered.params[0].args[0].args[0] == (2,)
    assert len(recovered.body) == 2
    kind, fresh = recovered.body[0]
    assert kind == "emit" and fresh.value == 23
    assert recovered.body[1] == ("return", recovered.params[0])
    index, line = next(
        (i, line) for i, line in enumerate(lines) if line.strip() == "M.record(M.node(23))"
    )
    column = line.index("M.node(23)") + 1
    span = fresh.span
    assert not isinstance(span, ir.SequentialSpan)
    assert span.source_name.name == __file__
    assert (span.line, span.column, span.end_line, span.end_column) == (
        start + index,
        column,
        start + index,
        column + len("M.node(23)"),
    )


def test_store_rhs_failure_precedes_target_evaluation(language):
    # A failing store RHS must preserve the original exception and never evaluate its target.
    M = language.M
    failure = ValueError("store value failed")
    effects = []

    def value():
        effects.append("value")
        raise failure

    def target():
        effects.append("target")
        return [0]

    with pytest.raises(ValueError) as caught:

        @M.function
        def main():
            target()[0] = value()

    assert caught.value is failure
    assert effects == ["value"]
    lines, first = inspect.getsourcelines(test_store_rhs_failure_precedes_target_evaluation)
    location = first + next(i for i, line in enumerate(lines) if line.strip() == "raise failure")
    assert any(
        frame.filename == __file__ and frame.lineno == location
        for frame in traceback.extract_tb(caught.value.__traceback__)
    )


def test_namespace_rebinding_reports_source_location(language):
    # A fixed source namespace cannot become a local binding; the error must identify that target.
    M = language.M
    with pytest.raises(SyntaxError, match="namespace|shadow|rebind") as caught:

        @M.function
        def main():
            M = 1

    error = caught.value
    lines, first = inspect.getsourcelines(test_namespace_rebinding_reports_source_location)
    line = first + next(i for i, text in enumerate(lines) if text.strip().startswith("M = 1"))
    assert (error.filename, error.lineno, error.end_lineno) == (__file__, line, line)
    assert (error.offset, error.end_offset) == (13, 14)
    assert not language.functions


@pytest.mark.parametrize(
    "name, statement",
    [
        ("range", "range = 1"),
        ("int", "int: object = 1"),
        ("M", "M += 1"),
        ("range", "(range, other) = (1, 2)"),
        ("int", "for int in M.grid(2):\n    pass"),
        ("M", "with M.grid(2) as M:\n    pass"),
        ("range", "def range():\n    pass"),
        ("int", "class int:\n    pass"),
        ("M", "import math as M"),
        ("range", "from math import floor as range"),
        ("int", "values = [0 for int in (1,)]"),
        ("M", "value = lambda M: 0"),
        ("range", "value = (range := 1)"),
        ("int", "del int"),
        ("M", "try:\n    pass\nexcept ValueError as M:\n    pass"),
        ("range", "match 1:\n    case range:\n        pass"),
        ("int", "match [1]:\n    case [*int]:\n        pass"),
        ("M", "match {}:\n    case {**M}:\n        pass"),
    ],
)
def test_reserved_bindings_fail_before_construction(language, name, statement):
    source = "@M.function\ndef main():\n    effect()\n" + "\n".join(
        "    " + line for line in statement.splitlines()
    )
    effects = []
    with pytest.raises(SyntaxError, match=repr(name)) as caught:
        entry.parse(
            source, {"M": language.M, "effect": lambda: effects.append(1)}, filename="reserved.py"
        )
    error = caught.value
    assert error.filename == "reserved.py"
    assert error.lineno >= 4
    assert name in source.splitlines()[error.lineno - 1]
    assert error.offset > 0
    assert not language.functions
    assert not effects


@pytest.mark.parametrize(
    "name, parameter", [("M", "M: object"), ("range", "*range"), ("int", "**int")]
)
def test_reserved_parameters_fail_before_construction(language, name, parameter):
    source = "@M.function\ndef main(" + parameter.format(name=name) + "):\n    pass"
    with pytest.raises(SyntaxError, match=repr(name)) as caught:
        entry.parse(source, {"M": language.M}, filename="parameter.py")
    assert (caught.value.filename, caught.value.lineno) == ("parameter.py", 2)
    assert source.splitlines()[1][caught.value.offset - 1 :].startswith(name)
    assert not language.functions


@pytest.mark.parametrize("name", ["range", "int"])
def test_reserved_captured_builtin_reports_its_use(language, name):
    source = f"@M.function\ndef main():\n    {name}(2)"
    with pytest.raises(SyntaxError, match=repr(name)) as caught:
        entry.parse(source, {"M": language.M, name: lambda value: value}, filename="capture.py")
    assert (caught.value.filename, caught.value.lineno, caught.value.offset) == ("capture.py", 3, 5)
    assert not language.functions


@pytest.mark.parametrize("binding", ["alias, *other = M, 1, 2", "value = (alias := M)"])
def test_local_namespace_alias_is_rejected(language, binding):
    source = "@M.function\ndef main():\n    " + binding
    with pytest.raises(SyntaxError, match="namespace 'M'.*alias") as caught:
        entry.parse(source, {"M": language.M}, filename="alias.py")
    assert (caught.value.filename, caught.value.lineno) == ("alias.py", 3)
    assert source.splitlines()[2][caught.value.offset - 1] == "M"
    assert not language.functions


@pytest.mark.parametrize(
    "prefix",
    ["class Module:", "if True:"],
)
def test_host_import_cannot_replace_reserved_names(language, prefix):
    effects = []
    source = prefix + "\n    from math import floor as range\n    effect()\n"
    source += (
        "    @M.function\n    def main():\n        pass"
        if prefix.startswith("class")
        else "@M.function\ndef main():\n    pass"
    )
    with pytest.raises(SyntaxError, match="'range'.*reserved") as caught:
        entry.parse(
            source, {"M": language.M, "effect": lambda: effects.append(1)}, filename="import.py"
        )
    assert (caught.value.filename, caught.value.lineno) == ("import.py", 2)
    assert not effects
    assert not language.functions


def test_undefined_name_reports_the_original_source(language):
    # A missing body value must remain a Python NameError at the user's identifier.
    M = language.M
    with pytest.raises(NameError, match="missing_value") as caught:

        @M.function
        def main():
            return missing_value  # noqa: F821

    assert type(caught.value) is NameError
    lines, first = inspect.getsourcelines(test_undefined_name_reports_the_original_source)
    index, line = next(
        (i, line) for i, line in enumerate(lines) if line.strip().startswith("return missing_value")
    )
    location = first + index
    frames = traceback.extract_tb(caught.value.__traceback__)
    source_frames = [
        frame for frame in frames if frame.filename == __file__ and frame.lineno == location
    ]
    assert source_frames
    if getattr(source_frames[-1], "colno", None) is not None:
        column = line.index("missing_value")
        assert (
            source_frames[-1].colno,
            source_frames[-1].end_lineno,
            source_frames[-1].end_colno,
        ) == (
            column,
            location,
            column + len("missing_value"),
        )
