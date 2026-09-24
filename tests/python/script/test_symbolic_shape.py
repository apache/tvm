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
"""Symbolic dimensions in function signatures share identity with body declarations.
Quoted dimensions must not create or replace ordinary Python bindings.
"""

from __future__ import annotations

# Invalid script examples deliberately contain unresolved or unused bindings.
# ruff: noqa: F821, F841
import inspect

import pytest

from tvm import ir
from tvm.script import tirx as T


def test_signature_symbols_cross_nested_calls_parameters_return_and_body(language):
    # Repeated quoted dimensions in nested annotations and body declarations must resolve to
    # the same symbol.
    M = language.M
    M.tuple = lambda *fields: fields

    @M.function
    def main(
        x: M.tuple(M.Tensor(("n",), "float32"), M.Tensor(("n",), "float32")),
        y: M.Tensor(("n",), "float32"),
    ) -> M.Tensor(("n",), "float32"):
        n = M.symbol()
        M.record(n)
        return y

    n = main.params[0].args[0][0].args[0][0]
    assert main.params[0].args[0][1].args[0][0] is n
    assert main.params[1].args[0].args[0][0] is n
    assert main.ret_type.args[0][0] is n
    assert main.body[0][1] is n


def test_signature_read_before_introduction_remains_unbound(language):
    # A quoted dimension must not introduce an ordinary Python name before its declaration.
    M = language.M
    with pytest.raises(NameError, match="n"):

        @M.function
        def main(x: M.Tensor((n, "n"), "float32")):
            return x


def test_signature_strings_do_not_replace_captured_python_names(language):
    # A captured Python dimension and a same-spelling quoted symbol must remain distinct.
    M = language.M
    n = 7

    @M.function
    def main(x: M.Tensor((n, "n"), "float32"), y: M.Tensor((n,), "float32")):
        M.record(n)
        return y

    first, symbol = main.params[0].args[0].args[0]
    assert first == 7
    assert main.params[1].args[0].args[0][0] == 7
    assert main.body[0][1] == 7
    assert symbol.name == "n"


def test_captured_shape_requires_concrete_symbols():
    # Captured tuples bypass literal decoding; native shape construction must
    # preserve concrete symbols and reject captured strings instead of inventing vars.
    def build(shape):
        @T.prim_func
        def main(x: T.Buffer(shape, "float32")):
            T.evaluate(0)

        return main

    n = ir.Var("n", "int64")
    function = build((n, 16))
    assert function.params[0].ty.shape[0].same_as(n)
    with pytest.raises(
        TypeError, match="^Builder expression arguments require concrete symbols, not strings$"
    ) as caught:
        build(("n", 16))
    assert type(caught.value) is TypeError


def _line_of(function, statement):
    lines, first = inspect.getsourcelines(function)
    return first + next(index for index, line in enumerate(lines) if line.strip() == statement)


def test_argument_policies_reuse_symbols_and_resolve_only_marked_literals(language):
    # Quoted shapes must reuse one symbol while only marked device strings resolve.
    M = language.M
    device = object()
    language.global_infos["cuda:1"] = device

    @M.function
    def main(x: M.Tensor(shape=("n + 1", "n"), dtype="float32", device="cuda:1")):
        M.record(x)

    annotation = main.params[0].args[0]
    shape, dtype, resolved, placement = annotation.args
    assert shape[0].op == "add" and shape[0].args[0] is shape[1]
    assert shape[0].args[1] == 1
    assert (dtype, placement) == ("float32", "S[0]")
    assert resolved is device
    assert main.body == [("emit", main.params[0])]


def test_quoted_symbols_do_not_introduce_python_bindings(language):
    # Resolving a quoted dimension must not silently define its unquoted body name.
    M = language.M
    with pytest.raises(NameError, match="n"):

        @M.function
        def main(x: M.Tensor(("n", "n"))):
            M.record(n)


def test_symbol_reassignment_reports_introduction_and_exact_write(language):
    # Ordinary writes to a symbolic dimension must point to its original introduction and
    # exact target.
    M = language.M
    with pytest.raises(SyntaxError) as caught:

        @M.function
        def main():
            n = M.symbol()
            n = 2

    message = str(caught.value)
    introduction = _line_of(
        test_symbol_reassignment_reports_introduction_and_exact_write, "n = M.symbol()"
    )
    offending = _line_of(test_symbol_reassignment_reports_introduction_and_exact_write, "n = 2")
    assert "Symbolic variable 'n' cannot be reassigned" in message
    assert f"introduced at line {introduction}" in message
    error = caught.value
    assert (error.filename, error.lineno, error.end_lineno) == (__file__, offending, offending)
    assert (error.offset, error.end_offset) == (13, 14)


def test_repeated_symbol_declarations_reuse_identity(language):
    # Repeated explicit declarations must reuse the annotation-owned symbol.
    M = language.M

    @M.function
    def main(x: M.Tensor(("n",))):
        n = M.symbol()
        M.record(n)
        n = M.symbol()
        M.record(n)

    symbol = main.params[0].args[0].args[0][0]
    assert main.body == [("emit", symbol), ("emit", symbol)]
    assert main.body[0][1] is main.body[1][1]


def test_nested_scope_does_not_reassign_outer_symbol(language):
    # An ordinary nested local must not overwrite an enclosing symbolic dimension.
    M = language.M

    @M.function
    def main():
        n = M.symbol()

        @M.function
        def nested():
            n = 2
            M.record(n)

        M.record(n)

    assert main.body[0][1].op == "symbol"
    assert language.functions["nested"].body == [("emit", 2)]
