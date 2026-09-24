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
import sys
from textwrap import dedent

from tvm.script.parser import entry


def test_parse_string_returns_fresh_symbols(language):
    # The direct string API must create independent symbol/parameter objects on repeated calls.
    # This is the suite's single dedicated parse(str) API case.
    source = '@M.function\ndef main(x: M.Tensor((M.dynamic("n") + 1,))):\n    M.record(x)\n'
    first = entry.parse(source, extra_vars={"M": language.M}, root_builder=language.M)
    second = entry.parse(source, extra_vars={"M": language.M}, root_builder=language.M)
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

    def identity(x: M.Tensor((4,))):
        M.record(x)

    programs = []
    recompose = entry._recompose_builder

    def observe(translated, **kwargs):
        programs.append(ast.unparse(translated))
        return recompose(translated, **kwargs)

    # Observe actual generated output; the original recomposition and execution still run.
    monkeypatch.setattr(entry, "_recompose_builder", observe)
    result = entry.parse(identity, root_builder=M, track_span=False)
    expected = dedent(
        """
        with _I0.IRBuilder() as _builder0:
            with M.function_() as _fn0:
                M.func_name('identity')
                x = M.arg('x', M.Tensor((4,)))

                def _build0():
                    x, = _fn0.params
                    M.emit_(M.record(x))
                _build0()
        _result0 = _fn0.function
        _result0.__name__ = 'identity'
        M.check_well_formed_(_result0)
        """
    ).strip()
    if sys.version_info[:2] == (3, 10):
        # Python 3.10's ast.unparse parenthesizes this tuple assignment target.
        expected = expected.replace("x, = _fn0.params", "(x,) = _fn0.params")
    assert programs == [expected]
    assert result.params[0].args[0].args[0] == (4,)
    assert result.body == [("emit", result.params[0])]
