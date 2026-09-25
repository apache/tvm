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
import traceback

import pytest

from tvm import ir
from tvm.ir import prim
from tvm.script import tirx as T
from tvm.script.parser import entry


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


@pytest.mark.parametrize(
    "declaration",
    [
        "def main(value: {annotation}):\n    pass",
        "def main(value: {annotation}, /):\n    pass",
        "def main(*, value: {annotation}):\n    pass",
        "def main(*values: {annotation}):\n    pass",
        "def main(**values: {annotation}):\n    pass",
        "def main() -> {annotation}:\n    pass",
        "def main():\n    value: {annotation} = effect()",
    ],
)
@pytest.mark.parametrize(
    "annotation",
    [
        repr('M.Tensor((4,), "float32")'),
        '"""M.Tensor(\n    (4,), "float32"\n)"""',
    ],
)
@pytest.mark.parametrize("decorator", ["function", "inline"])
def test_quoted_source_annotations_report_the_complete_literal(
    spanned_language, declaration, annotation, decorator
):
    language = spanned_language
    source = f"from __future__ import annotations\n@M.{decorator}\n" + declaration.format(
        annotation=annotation
    )
    literal = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant))
    effects = []
    with pytest.raises(SyntaxError, match="Quoted annotations are not supported") as caught:
        entry.parse(
            source,
            {"M": language.M, "effect": lambda: effects.append(1)},
            filename="quoted_annotation.py",
        )
    error = caught.value
    assert (
        error.filename,
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    ) == (
        "quoted_annotation.py",
        literal.lineno,
        literal.col_offset + 1,
        literal.end_lineno,
        literal.end_col_offset + 1,
    )
    assert not effects
    assert not language.functions


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
