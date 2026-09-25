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
"""Public string and callable entries build fresh script objects.

A small spans-off callable also documents the generated builder program exactly.
"""

from __future__ import annotations

# Keep the single generated-program golden exactly as ast.unparse emits it.
import ast
from textwrap import dedent

from tvm.script.parser import entry


def test_parse_string_returns_fresh_symbols(language):
    # The direct string API must create independent symbol/parameter objects on repeated calls.
    # This is the suite's single dedicated parse(str) API case.
    source = '@M.function\ndef main(x: M.Tensor((M.dynamic("n") + 1,))):\n    M.record(x)\n'
    first = entry.parse(source, extra_vars={"M": language.M})
    second = entry.parse(source, extra_vars={"M": language.M})
    assert first.params[0] is not second.params[0]
    first_shape = first.params[0].args[0].args[0][0]
    second_shape = second.params[0].args[0].args[0][0]
    assert first_shape.op == second_shape.op == "add"
    assert first_shape.args[0] is not second_shape.args[0]
    assert first_shape.args[1] == second_shape.args[1] == 1
    assert first.body[0][1] is first.params[0]
    assert second.body[0][1] is second.params[0]


def test_callable_entry_emits_the_expected_builder_program(language, monkeypatch):
    # Normal callable parsing must construct the documented builder program without span
    # scaffolding.
    M = language.M

    programs = []
    recompose = entry._recompose_builder

    def observe(translated, **kwargs):
        programs.append(ast.unparse(translated))
        return recompose(translated, **kwargs)

    # Observe actual generated output; the original recomposition and execution still run.
    monkeypatch.setattr(entry, "_recompose_builder", observe)
    parse = entry.parse

    def without_spans(*args, **kwargs):
        return parse(*args, track_span=False, **kwargs)

    monkeypatch.setattr(entry, "parse", without_spans)

    @M.function
    def identity(x: M.Tensor((4,))):
        M.record(x)

    result = identity
    expected = dedent(
        """
        with _I0.IRBuilder() as _builder0:
            _definition0 = _host2(('M',), _host0(), _host1())
            with M.function_() as _fn0:

                def _declare0(*, M=_definition0.get('M', _I0.MISSING)):
                    M.func_name_('identity')
                    x = M.arg_('x', M.Tensor((4,)))

                _declare0()

                def _build0():
                    x, = _fn0.params
                    M.emit_(M.record(x))
                _build0()
        _result0 = _fn0.function
        _result0.__name__ = 'identity'
        M.check_well_formed_(_result0)
        """
    ).strip()
    assert len(programs) == 1
    assert ast.dump(ast.parse(programs[0])) == ast.dump(ast.parse(expected))
    assert result.params[0].args[0].args[0] == (4,)
    assert result.body == [("emit", result.params[0])]


def test_source_prefix_and_lazy_annotation_capture(language):
    source = """
extent = 3
@M.function
def main(x: M.Tensor((missing if I.constexpr(False) else extent,))):
    extent = 5
    M.record(extent)
"""
    result = entry.parse(source, extra_vars={"M": language.M}, definition_scope={"extent": 128})
    assert result.params[0].args[0].args[0] == (3,)
    assert result.body == [("emit", 5)]


def test_text_definition_defaults_preserve_body_globals_and_prefix_closures(language):
    source = """
@M.function
def main(x: M.Tensor((extent,))):
    def nested():
        return extent
    M.record(nested())
"""
    result = entry.parse(
        source, extra_vars={"M": language.M, "extent": 256}, definition_scope={"extent": 128}
    )
    assert result.params[0].args[0].args[0] == (128,)
    assert result.body == [("emit", 256)]

    prefixed = entry.parse(
        "extent = 3\n" + source,
        extra_vars={"M": language.M, "extent": 256},
        definition_scope={"extent": 128},
    )
    assert prefixed.params[0].args[0].args[0] == (3,)
    assert prefixed.body == [("emit", 3)]


def test_module_member_retains_completed_function():
    from tvm.script import tirx as T

    @T.prim_func(private=True)
    def completed():
        T.evaluate(7)

    module = entry.parse(
        """@I.ir_module
class Module:
    helper = completed
""",
        extra_vars={"completed": completed},
    )
    assert module["helper"].same_as(completed)
    assert module["helper"].body.value.value == 7


def test_python_helpers_retain_quoted_annotations(language):
    module = entry.parse(
        "\n".join(
            [
                "@I.ir_module",
                "class Module:",
                "    @I.pyfunc",
                '    def helper(value: "int") -> "int":',
                '        def nested(item: "int") -> "int":',
                "            return item",
                "        return nested(value)",
            ]
        )
    )
    helper = module.__pyfuncs__["helper"]
    assert helper(7) == 7
    assert helper.__annotations__ == {"value": "int", "return": "int"}
    function = entry.parse(
        "\n".join(
            [
                "@M.function",
                "def main():",
                '    def helper(value: "int") -> "int":',
                "        return value",
                "    M.record(helper(5))",
            ]
        ),
        extra_vars={"M": language.M},
    )
    assert function.body == [("emit", 5)]
