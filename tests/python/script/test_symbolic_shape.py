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
"""Explicit symbolic dimensions retain identity in signatures and function bodies.
Dimensions use ordinary Python expressions over explicit symbols.
"""

from __future__ import annotations

# Invalid script examples deliberately contain unresolved or unused bindings.
# ruff: noqa: F821
import sys

import pytest

from tvm import ir
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.parser import entry


def test_signature_symbols_cross_nested_calls_parameters_return_and_body(language):
    # Nested annotations and body reads retain the externally constructed symbol.
    M = language.M
    M.tuple = lambda *fields: fields
    n = M.dynamic("n")

    @M.function
    def main(
        x: M.tuple(M.Tensor((n,), "float32"), M.Tensor((n,), "float32")),
        y: M.Tensor((n,), "float32"),
    ) -> M.Tensor((n,), "float32"):
        M.record(n)
        return y

    n = main.params[0].args[0][0].args[0][0]
    assert main.params[0].args[0][1].args[0][0] is n
    assert main.params[1].args[0].args[0][0] is n
    assert main.ret_type.args[0][0] is n
    assert main.body[0][1] is n


def test_signature_read_before_introduction_remains_unbound(language):
    # An undeclared dimension must remain an ordinary unbound Python name.
    M = language.M
    with pytest.raises(NameError, match="n"):

        @M.function
        def main(x: M.Tensor((n,), "float32")):
            return x


def test_same_named_symbol_does_not_replace_captured_python_value(language):
    # A captured Python dimension and a same-spelling external symbol remain distinct.
    M = language.M
    n = 7
    symbol = M.dynamic("n")

    @M.function
    def main(x: M.Tensor((n, symbol), "float32"), y: M.Tensor((n,), "float32")):
        M.record(n)
        return y

    first, symbol = main.params[0].args[0].args[0]
    assert first == 7
    assert main.params[1].args[0].args[0][0] == 7
    assert main.body[0][1] == 7
    assert symbol.name == "n"


def test_captured_shape_requires_concrete_symbols():
    # Native shape construction preserves concrete symbols and rejects strings.
    def build(shape):
        @T.prim_func
        def main(x: T.Buffer(shape, "float32")):
            T.evaluate(0)

        return main

    n = T.dynamic("n")
    function = build((n, 16))
    assert function.params[0].ty.shape[0].same_as(n)
    with pytest.raises(AssertionError, match="data must be int or Expr, but got n"):
        build(("n", 16))


def test_argument_policies_reuse_symbols_and_resolve_only_marked_literals(language):
    # Shape expressions reuse the external symbol while marked device strings resolve.
    M = language.M
    device = object()
    language.global_infos["cuda:1"] = device
    n = M.dynamic("n")

    @M.function
    def main(x: M.Tensor(shape=(n + 1, n), dtype="float32", device="cuda:1")):
        M.record(x)

    annotation = main.params[0].args[0]
    shape, dtype, resolved, placement = annotation.args
    assert shape[0].op == "add" and shape[0].args[0] is shape[1]
    assert shape[0].args[1] == 1
    assert (dtype, placement) == ("float32", "S[0]")
    assert resolved is device
    assert main.body == [("emit", main.params[0])]


def test_external_symbol_does_not_introduce_same_named_python_binding(language):
    # Capturing a symbol under another name must not introduce its IR name in Python.
    M = language.M
    symbol = M.dynamic("n")
    with pytest.raises(NameError, match="n"):

        @M.function
        def main(x: M.Tensor((symbol, symbol))):
            M.record(n)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_symbol_reassignment_reports_introduction_and_exact_write(language):
    # Ordinary writes to a header symbol must point to its declaration and exact target.
    source = """
@M.function
def main[n](x: M.Tensor((n,))):
    n = 2
"""
    filename = "symbol_reassignment.py"
    with pytest.raises(SyntaxError) as caught:
        entry.parse(source, extra_vars={"M": language.M}, filename=filename)

    message = str(caught.value)
    introduction = 3
    offending = 4
    assert "Symbolic variable 'n' cannot be reassigned" in message
    assert f"introduced at line {introduction}" in message
    error = caught.value
    assert (error.filename, error.lineno, error.end_lineno) == (filename, offending, offending)
    assert (error.offset, error.end_offset) == (5, 6)


def test_external_dynamic_symbols_reuse_identity(language):
    # Repeated captures share the exact externally constructed symbol.
    M = language.M

    n = M.dynamic("n")

    @M.function
    def main(x: M.Tensor((n,))):
        M.record(n)
        M.record(n)

    symbol = main.params[0].args[0].args[0][0]
    assert main.body == [("emit", symbol), ("emit", symbol)]
    assert main.body[0][1] is main.body[1][1]


def test_nested_scope_does_not_reassign_outer_symbol(language):
    # An ordinary nested local must not overwrite an enclosing symbolic dimension.
    M = language.M

    n = M.dynamic("n")

    @M.function
    def main():
        @M.function
        def nested():
            n = 2
            M.record(n)

        M.record(n)

    assert main.body[0][1].op == "symbol"
    assert language.functions["nested"].body == [("emit", 2)]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_mixed_generic_symbol_dtypes_and_identity(language):
    source = """
@M.function
def main[n, k: M.int32](x: M.Tensor((n, k))) -> M.Tensor((n, k)):
    M.record(n)
    M.record(k)
    return x
"""
    main = entry.parse(source, extra_vars={"M": language.M})
    n, k = main.params[0].args[0].args[0]
    assert n.args == ("int64",)
    assert k.args == ("int32",)
    assert main.ret_type.args[0][0] is n
    assert main.ret_type.args[0][1] is k
    assert main.body[0][1] is n
    assert main.body[1][1] is k


def test_dynamic_symbols_are_fresh_and_scope_independent():
    assert T.dynamic is I.dynamic
    n = T.dynamic("n")
    same_name = I.dynamic("n")
    k = I.dynamic("k", "int32")
    assert n.ty.dtype == "int64"
    assert k.ty.dtype == "int32"
    assert not n.same_as(same_name)

    @I.ir_module
    class Module:
        @T.prim_func
        def first(x: T.Buffer((n,), "float32")):
            T.evaluate(n)

    assert Module["first"].params[0].ty.shape[0].same_as(n)
    assert Module["first"].body.value.same_as(n)
