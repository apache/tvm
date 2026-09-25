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
"""Specialize real JIT scripts while retaining runtime parameters and captured values.

A small native TIRx example checks the resulting IR. Mini-language regressions
cover deferred annotations, nested failures and the lifetime of needed captures.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.parser import protocol_registry
from tvm.tirx.script.jit import make_jit


@pytest.fixture
def jit_language(language):
    language.M.jit = protocol_registry.declaration_kind("M.jit", "function")(make_jit(language.M))
    return language


def test_annotation_constructor_executes_in_required_builder_context(jit_language):
    # Before: a postponed annotation calls a context-sensitive constructor.
    # Expected builder: constructor runs once at specialization inside the real IRBuilder.
    M = jit_language.M
    calls = []

    def annotation():
        calls.append(IRBuilder.is_in_scope())
        return M.Tensor((5,), "int32")

    @M.jit
    def function(output: annotation()):
        M.record(7)

    assert calls == []  # postponed annotations are not evaluated at decoration
    result = function.specialize()
    assert calls == [True]
    assert [int(value) for value in result.params[0].args[0].args[0]] == [5]


def test_optional_annotation_is_evaluated_only_for_present_parameters(jit_language):
    # A selected absence skips its annotation entirely; a present optional unwraps once.
    M = jit_language.M
    M.Optional = T.Optional
    calls = []

    def annotation():
        calls.append("annotation")
        return M.Tensor((5,))

    @M.jit
    def function(value: M.Optional(annotation())):
        M.record(value)

    absent = function.specialize(value=None)
    assert absent.params == [] and absent.body == [("emit", None)]
    assert calls == []
    present = function.specialize()
    assert calls == ["annotation"]
    assert present.params[0].args[0].args[0] == (5,)
    assert present.body == [("emit", present.params[0])]


def test_jit_retains_lexical_annotation_snapshot(jit_language):
    # Later enclosing mutations must not replace the annotation/body values captured for JIT.
    M = jit_language.M

    def outer(extent):
        def middle(width):
            type_namespace = M

            @M.jit
            def function(output: type_namespace.Tensor((extent, width), "int32")):
                M.record(extent)

            return function

        pending = middle(3)
        extent = 99
        return pending

    function = outer(8).specialize()
    assert function.params[0].args[0].args[:2] == ((8, 3), "int32")
    assert function.body == [("emit", 8)]


def test_reentrant_specialization_restores_root_bindings_after_failure(jit_language):
    # A nested parse and failed same-name specialization must not overwrite the outer value.
    M = jit_language.M
    failure = ValueError("nested failure")

    def fail():
        raise failure

    def nested():
        @M.function
        def kernel(n: M.Tensor((1,))):
            M.record(n)

        assert len(kernel.params) == 1
        assert kernel.body[0][1] is kernel.params[0]

        @M.jit
        def kernel(n: I.constexpr):
            fail()

        with pytest.raises(ValueError) as caught:
            kernel.specialize(n=8)
        assert caught.value is failure

    @M.jit
    def kernel(n: I.constexpr):
        nested()
        M.record(n)

    result = kernel.specialize(n=4)
    assert result.params == [] and result.body[-1] == ("emit", 4)


def test_live_jit_retains_only_needed_scope_across_uncached_builds(jit_language):
    # A live JIT must release unrelated scope while preserving annotations across uncached builds.
    class Payload:
        pass

    M = jit_language.M
    M.jit = protocol_registry.declaration_kind("M.jit", "function")(make_jit(M))

    def make():
        payload = Payload()
        reference = weakref.ref(payload)
        width = 7

        @M.jit
        def kernel(x: M.Tensor((width,)), *, value: I.constexpr):
            M.record(value)

        return kernel, reference

    enabled = gc.isenabled()
    gc.disable()
    try:
        kernel, reference = make()
        assert reference() is None
        first = kernel.specialize(value=1)
        second = kernel.specialize(value=2)
        assert reference() is None
        assert first is kernel.specialize(value=1) and first is not second
        assert first.params[0].args[0].args[0] == second.params[0].args[0].args[0] == (7,)
        assert first.body == [("emit", 1)] and second.body == [("emit", 2)]
    finally:
        if enabled:
            gc.enable()


def test_recursive_parameters_survive_empty_specialization(jit_language):
    # Empty specialization must preserve recursive runtime arguments and their identities.
    M = jit_language.M
    M.jit = protocol_registry.declaration_kind("M.jit", "function")(make_jit(M))

    @M.jit
    def main(x: M.Tensor((4,)), y: M.Tensor((4,))):
        main(x, y)

    result = main.specialize()
    assert len(result.params) == 2
    call = result.body[0][1]
    assert call.args[0] is jit_language.references["main"]
    assert all(actual is expected for actual, expected in zip(call.args[1:], result.params))
    assert len(call.args) == 3


def test_tirx_jit_specializes_captured_shape_and_value():
    # A deferred native function must retain its enclosing shape and substitute the constexpr value.
    width = 4

    @T.jit(private=True)
    def fill(output: T.Buffer((width,), "int32"), *, value: T.constexpr):
        for index in range(width):
            output[index] = value

    @T.prim_func(private=True)
    def expected(output: T.Buffer((4,), "int32")):
        for index in range(4):
            output[index] = 3

    result = fill.specialize(value=3)
    assert len(result.params) == 1
    assert_structural_equal(result, expected, map_free_vars=True)


def test_optional_missing_annotation_stays_lazy(jit_language):
    M = jit_language.M
    M.Optional = T.Optional

    @M.jit
    def function(value: M.Optional(missing_annotation)):  # noqa: F821
        M.record(value)

    absent = function.specialize(value=None)
    assert absent.params == [] and absent.body == [("emit", None)]
    with pytest.raises(NameError, match="missing_annotation"):
        function.specialize()


def test_jit_preserves_namespace_used_only_by_decorator(jit_language):
    Alias = jit_language.M

    @Alias.jit(private=True)
    def function():
        pass

    result = function.specialize()
    assert result.params == [] and result.body == []
