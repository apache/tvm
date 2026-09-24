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
"""Construction failures preserve the original error and allow later scripts to build.

A nested function failure must unwind lexical and construction scopes without
leaking its source context into a subsequently constructed native expression.
"""

import ast
import inspect
import sys
import traceback

import pytest

from tvm import ir
from tvm.ir import prim
from tvm.script import tirx as T
from tvm.script.parser import entry


@pytest.mark.parametrize(
    "definition, message, node_kind, version",
    [
        pytest.param(
            "def main(*args):\n    pass",
            "ordinary named parameters",
            ast.FunctionDef,
            (3, 10),
            id="varargs",
        ),
        pytest.param(
            "def main(**kwargs):\n    pass",
            "ordinary named parameters",
            ast.FunctionDef,
            (3, 10),
            id="kwargs",
        ),
        pytest.param(
            "def main():\n    try:\n        pass\n    finally:\n        pass",
            "Unsupported statement: Try",
            ast.Try,
            (3, 10),
            id="try",
        ),
        pytest.param(
            "def main():\n    raise ValueError()",
            "Unsupported statement: Raise",
            ast.Raise,
            (3, 10),
            id="raise",
        ),
        pytest.param(
            "def main():\n    del value",
            "Unsupported statement: Delete",
            ast.Delete,
            (3, 10),
            id="delete",
        ),
        pytest.param(
            "def main():\n    match 0:\n        case 0:\n            pass",
            "Unsupported statement: Match",
            ast.Match,
            (3, 10),
            id="match",
        ),
        pytest.param(
            "def main[n: float]():\n    pass",
            "bound must be int",
            "TypeVar",
            (3, 12),
            id="type-bound",
        ),
        pytest.param(
            "def main[*ns]():\n    pass",
            "Only scalar type parameters",
            "TypeVarTuple",
            (3, 12),
            id="type-tuple",
        ),
        pytest.param(
            "def main[**ps]():\n    pass",
            "Only scalar type parameters",
            "ParamSpec",
            (3, 12),
            id="param-spec",
        ),
        pytest.param(
            "def main[n = int]():\n    pass",
            "cannot have a default",
            "TypeVar",
            (3, 13),
            id="type-default",
        ),
    ],
)
def test_rejected_syntax_keeps_source_range_and_recovers(
    language, definition, message, node_kind, version
):
    if sys.version_info < version:
        pytest.skip(f"requires Python {version[0]}.{version[1]} syntax")
    source = "@M.function\n" + definition + "\n"
    kind = getattr(ast, node_kind) if isinstance(node_kind, str) else node_kind
    node = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, kind))
    with pytest.raises(SyntaxError, match=message) as caught:
        entry.parse(source, extra_vars={"M": language.M}, filename="rejected.py")
    error = caught.value
    assert type(error) is SyntaxError
    assert (error.filename, error.lineno, error.offset, error.end_lineno, error.end_offset) == (
        "rejected.py",
        node.lineno,
        node.col_offset + 1,
        node.end_lineno,
        node.end_col_offset + 1,
    )
    assert not language.functions
    result = entry.parse("@M.function\ndef valid():\n    pass\n", extra_vars={"M": language.M})
    assert result.name == "valid"


def test_named_parameter_kinds_remain_supported(language):
    result = entry.parse(
        "@M.function\ndef main(x: M.Tensor((4,)), /, *, y: M.Tensor((4,))):\n    pass\n",
        extra_vars={"M": language.M},
    )
    assert [parameter.name for parameter in result.params] == ["x", "y"]


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


def test_optional_annotation_requires_jit_at_the_source_parameter():
    # Ordinary argument validation must reject Optional at its original parameter location.
    with pytest.raises(TypeError, match="^T.Optional is only supported by @T.jit$") as caught:

        @T.prim_func
        def invalid(value: T.Optional(T.handle)):
            T.evaluate(0)

    assert type(caught.value) is TypeError
    lines, first = inspect.getsourcelines(
        test_optional_annotation_requires_jit_at_the_source_parameter
    )
    index, line = next((i, text) for i, text in enumerate(lines) if "def invalid(" in text)
    frames = traceback.extract_tb(caught.value.__traceback__)
    source = [
        frame for frame in frames if frame.filename == __file__ and frame.lineno == first + index
    ]
    assert source
    if getattr(source[-1], "colno", None) is not None:
        column = line.index("value:")
        assert (source[-1].colno, source[-1].end_lineno, source[-1].end_colno) == (
            column,
            first + index,
            column + len("value: T.Optional(T.handle)"),
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
            M = 1  # noqa: F841

    error = caught.value
    lines, first = inspect.getsourcelines(test_namespace_rebinding_reports_source_location)
    line = first + next(i for i, text in enumerate(lines) if text.strip().startswith("M = 1"))
    assert (error.filename, error.lineno, error.end_lineno) == (__file__, line, line)
    assert (error.offset, error.end_offset) == (13, 14)
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


def test_invalid_quoted_annotation_keeps_source_range(language):
    # A malformed quoted annotation must report the original literal, not generated code.
    M = language.M
    with pytest.raises(SyntaxError, match="Invalid annotation expression") as caught:

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
