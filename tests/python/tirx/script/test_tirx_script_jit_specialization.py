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
"""Native JIT annotation, capture, lifetime and specialization contracts."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest

from tvm import ir, tirx
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder


def test_jit_annotation_context_and_optional_selection():
    calls = []

    def annotation():
        calls.append(IRBuilder.is_in_scope())
        return T.Buffer((5,), "int32")

    @T.jit(private=True)
    def required(output: annotation()):
        output[0] = 7

    assert calls == []
    result = required.specialize()
    assert calls == [True]
    assert [int(value) for value in result.params[0].ty.shape] == [5]
    assert result.body.buffer.same_as(result.params[0])
    assert int(result.body.value) == 7
    calls.clear()

    @T.jit(private=True)
    def optional(value: T.Optional(annotation())):
        if T.constexpr(value is not None):
            value[0] = 3
        else:
            T.evaluate(0)

    assert calls == []
    absent = optional.specialize(value=None)
    assert len(absent.params) == 0 and int(absent.body.value) == 0
    assert calls == []
    present = optional.specialize()
    assert calls == [True]
    assert [int(value) for value in present.params[0].ty.shape] == [5]
    assert present.body.buffer.same_as(present.params[0])
    assert int(present.body.value) == 3

    @T.jit(private=True)
    def missing(value: T.Optional(missing_annotation)):  # noqa: F821
        T.evaluate(0)

    absent = missing.specialize(value=None)
    assert len(absent.params) == 0 and int(absent.body.value) == 0
    with pytest.raises(NameError, match="missing_annotation"):
        missing.specialize()


def test_jit_capture_snapshot_and_specialization():
    def outer(extent):
        def middle(width):
            type_namespace = T

            @T.jit(private=True)
            def function(output: type_namespace.Buffer((extent, width), "int32")):
                T.evaluate(extent)

            return function

        pending = middle(3)
        extent = 99
        return pending

    function = outer(8).specialize()
    assert [int(value) for value in function.params[0].ty.shape] == [8, 3]
    assert function.params[0].ty.dtype == "int32"
    assert int(function.body.value) == 8

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


def test_jit_reentrant_specialization_after_failure():
    failure = ValueError("nested failure")

    def fail():
        raise failure

    def nested():
        @T.prim_func(private=True)
        def kernel(n: T.int32):
            T.evaluate(n)

        assert len(kernel.params) == 1
        assert kernel.body.value.same_as(kernel.params[0])

        @T.jit(private=True)
        def kernel(n: I.constexpr):
            fail()

        with pytest.raises(ValueError) as caught:
            kernel.specialize(n=8)
        assert caught.value is failure

    @T.jit(private=True)
    def kernel(n: I.constexpr):
        nested()
        T.evaluate(n)

    result = kernel.specialize(n=4)
    assert len(result.params) == 0 and int(result.body.value) == 4


def test_jit_capture_lifetime_and_cache():
    class Payload:
        pass

    def make():
        payload = Payload()
        reference = weakref.ref(payload)
        width = 7

        @T.jit(private=True)
        def kernel(x: T.Buffer((width,), "int32"), *, value: I.constexpr):
            T.evaluate(value)

        return kernel, reference

    class Owner:
        def decorate(self, function):
            return function

    def decorated():
        owner = Owner()
        reference = weakref.ref(owner)

        @T.jit(private=True)
        @owner.decorate
        def function():
            pass

        return function, reference

    enabled = gc.isenabled()
    gc.disable()
    try:
        kernel, reference = make()
        assert reference() is None
        first = kernel.specialize(value=1)
        second = kernel.specialize(value=2)
        assert reference() is None
        assert first is kernel.specialize(value=1) and first is not second
        assert [int(value) for value in first.params[0].ty.shape] == [7]
        assert [int(value) for value in second.params[0].ty.shape] == [7]
        assert int(first.body.value) == 1 and int(second.body.value) == 2
        function, reference = decorated()
        assert reference() is None
        result = function.specialize()
        assert len(result.params) == 0
        assert_structural_equal(result.body, tirx.Evaluate(0))
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("annotation_form", ["eager", "postponed", "quoted"])
def test_jit_namespace_and_marker_resolution(tmp_path, monkeypatch, annotation_form):
    Alias = T

    @Alias.jit(private=True)
    def empty():
        pass

    result = empty.specialize()
    assert len(result.params) == 0
    assert "global_symbol" not in result.attrs
    assert_structural_equal(result.body, tirx.Evaluate(0))

    source = """@Script.jit(private=True)
def kernel(output: Script.Buffer((1,), "int32"), *, value: Script.constexpr,
           optional: Script.Optional(Script.Buffer((1,), "int32"))):
    output[0] = value
"""
    if annotation_form != "eager":
        source = "from __future__ import annotations\n" + source
    if annotation_form == "quoted":
        source = source.replace("value: Script.constexpr", "value: 'Script.constexpr'")
    path = tmp_path / "aliased_jit.py"
    path.write_text(source)
    namespace = {"Script": T}
    exec(compile(source, str(path), "exec", dont_inherit=True), namespace)
    kernel = namespace["kernel"]
    if annotation_form == "quoted":
        with pytest.raises(SyntaxError, match="Quoted annotations are not supported"):
            kernel.specialize(value=3, optional=None)
        return
    assert kernel.constexpr_names == {"value"}
    assert kernel.optional_names == {"optional"}
    result = kernel.specialize(value=3, optional=None)
    assert len(result.params) == 1 and int(result.body.value) == 3

    # Registered namespace syntax, rather than the current attribute value, selects markers.
    monkeypatch.setattr(T, "constexpr", object())
    monkeypatch.setattr(T, "marker_alias", I.constexpr, raising=False)
    Ordinary = SimpleNamespace(constexpr=I.constexpr, Optional=T.Optional)

    def unavailable():
        pytest.fail("Unselected annotations must remain unevaluated")

    @T.jit(private=True)
    def specialized(value: Alias.constexpr):
        T.evaluate(value)

    assert int(specialized.specialize(value=4).body.value) == 4

    @T.jit(private=True)
    def ordinary(
        value: Ordinary.constexpr,
        optional: Ordinary.Optional(unavailable()),
        alias: T.marker_alias,
    ):
        T.evaluate(value)

    assert ordinary.constexpr_names == ordinary.optional_names == set()


def test_jit_empty_specialization_preserves_runtime_parameters():
    @T.jit(private=True)
    def main(x: T.int32, y: T.int32):
        main(x, y)
        main(x, y)

    result = main.specialize()
    assert len(result.params) == 2
    assert len(result.body.seq) == 2
    first, second = [node.value for node in result.body.seq]
    assert isinstance(first.op, ir.GlobalVar) and first.op.name_hint == "main"
    assert first.op.same_as(second.op)
    for call in (first, second):
        assert len(call.args) == 2
        assert all(actual.same_as(expected) for actual, expected in zip(call.args, result.params))
