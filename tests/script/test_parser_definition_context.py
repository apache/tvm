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
"""Generated builders preserve definition annotations and Python lexical bodies."""

from __future__ import annotations

import numpy as np
import pytest

import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder

EXTENT = 11
VALUE = 1


def _delayed(n, decorator):
    @decorator
    def function(output: T.Buffer((n,), "int32")):
        n = 2
        output[0] = n + VALUE

    return function


@pytest.mark.parametrize("factory", [False, True])
def test_reused_jit_decorator_captures_application_scope(factory):
    decorator = T.jit() if factory else T.jit
    pending = _delayed(8, decorator)
    n = 99  # noqa: F841
    VALUE = 20  # noqa: F841
    function = pending.specialize()
    assert [int(value) for value in function.params[0].ty.shape] == [8]
    compiled = tvm.compile(function, target="llvm", tir_pipeline="tirx")
    output = tvm.runtime.tensor(np.zeros(8, dtype="int32"))
    compiled(output)
    np.testing.assert_array_equal(output.numpy(), [3, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("factory", [False, True])
def test_reused_eager_decorator_captures_application_scope(factory):
    decorator = T.prim_func() if factory else T.prim_func
    function = _delayed(4, decorator)
    assert [int(value) for value in function.params[0].ty.shape] == [4]


def test_ir_module_class_annotation_scope_does_not_replace_body_global():
    @I.ir_module
    class Module:
        EXTENT = 3

        @T.prim_func
        def function(output: T.Buffer((EXTENT,), "int32")):
            output[0] = EXTENT

    function = Module["function"]
    assert [int(value) for value in function.params[0].ty.shape] == [3]
    assert function.body.value.value == 11


def test_ordinary_class_annotation_scope_does_not_replace_body_global():
    class Holder:
        EXTENT = 3

        @T.prim_func
        def function(output: T.Buffer((EXTENT,), "int32")):
            output[0] = EXTENT

    assert [int(value) for value in Holder.function.params[0].ty.shape] == [3]
    assert Holder.function.body.value.value == 11


def test_jit_annotation_snapshot_precedes_later_closure_mutation():
    def build():
        n = 4

        @T.jit
        def function(output: T.Buffer((n,), "int32")):
            output[0] = n

        n = 9
        return function

    function = build().specialize()
    assert [int(value) for value in function.params[0].ty.shape] == [4]
    assert function.body.value.value == 4


def test_annotation_constructor_executes_in_required_builder_context():
    calls = []

    def annotation():
        calls.append(IRBuilder.is_in_scope())
        return T.Buffer((5,), "int32")

    @T.jit
    def function(output: annotation()):
        output[0] = 7

    assert calls == []  # postponed annotations are not evaluated at decoration
    result = function.specialize()
    assert calls == [True]
    assert [int(value) for value in result.params[0].ty.shape] == [5]


def test_enclosing_namespace_used_only_by_annotation_is_retained():
    def build():
        dtype_namespace = T

        @T.jit
        def function(output: dtype_namespace.Buffer(("n",), "int32")):
            output[0] = 7

        return function

    result = build().specialize()
    assert isinstance(result.params[0].ty.shape[0], tvm.ir.Var)
    assert result.params[0].ty.shape[0].name == "n"


def test_attribute_name_does_not_turn_a_closure_binding_into_a_global():
    from types import SimpleNamespace

    dtype = "int32"
    descriptor = SimpleNamespace(dtype="int64")

    @T.prim_func
    def function():
        T.evaluate(T.cast(3, dtype))
        T.evaluate(T.cast(4, descriptor.dtype))

    first, second = function.body.seq
    assert first.value.ty.dtype == "int32"
    assert first.value.value == 3
    assert second.value.ty.dtype == "int64"
    assert second.value.value == 4


def test_local_annotation_retains_names_absent_from_signature():
    def build(shape, dtype):
        @R.function
        def function(x: R.Tensor((2,), "float32")):
            y: R.Tensor(shape, dtype) = R.add(x, x)
            return y

        return function

    @R.function
    def expected(x: R.Tensor((2,), "float32")):
        y: R.Tensor((2,), "float32") = R.add(x, x)
        return y

    actual = build((2,), "float32")
    tvm.ir.assert_structural_equal(
        actual.with_attr("global_symbol", ""), expected.with_attr("global_symbol", "")
    )


def test_local_annotation_and_signature_share_retained_context():
    def build(shape, dtype):
        @I.ir_module
        class Module:
            @R.function
            def main(x: R.Tensor(shape, dtype)):
                with R.dataflow():
                    y: R.Tensor(shape, dtype) = R.add(x, x)
                    R.output(y)
                return y

        return Module

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2,), "float32")):
            with R.dataflow():
                y: R.Tensor((2,), "float32") = R.add(x, x)
                R.output(y)
            return y

    tvm.ir.assert_structural_equal(build((2,), "float32"), Expected)


def test_local_annotation_preserves_explicit_parameter_and_d3_symbols():
    dtype = "float32"

    @R.function
    def function(n: R.Prim("int64"), x: R.Tensor((n, "m"), "float32")):
        m = T.int64()
        y: R.Tensor((n, m), dtype) = R.add(x, x)
        return y

    n, x = function.params
    binding = function.body.blocks[0].bindings[0]
    assert binding.var.ty.shape[0].same_as(n)
    assert binding.var.ty.shape[1].same_as(x.ty.shape[1])
    assert function.ret_ty.shape[0].same_as(n)
    assert function.ret_ty.shape[1].same_as(x.ty.shape[1])


def test_local_annotation_keeps_rhs_before_constructor_evaluation():
    events = []

    def annotation():
        events.append("annotation")
        assert IRBuilder.is_in_scope()
        return R.Tensor((2,), "float32")

    def value(x):
        events.append("rhs")
        return R.add(x, x)

    @R.function
    def function(x: R.Tensor((2,), "float32")):
        y: annotation() = value(x)
        return y

    assert events == ["rhs", "annotation"]
    assert function.ret_ty == R.Tensor((2,), "float32")


def test_local_annotation_scope_does_not_overwrite_ordinary_body_global():
    dtype = "float32"

    @I.ir_module
    class Module:
        EXTENT = 3

        @R.function
        def main(x: R.Tensor((3,), "float32")):
            R.func_attr({"body_extent": EXTENT})
            y: "R.Tensor((EXTENT,), dtype)" = R.add(x, x)  # noqa: UP037
            return y

    function = Module["main"]
    assert function.attrs["body_extent"] == 11
    assert function.ret_ty.shape[0] == 3
    assert function.ret_ty.dtype == "float32"


def test_local_annotation_preserves_lambda_and_comprehension_bindings():
    dtype = "int64"  # noqa: F841
    size = 9  # noqa: F841
    sizes = (2,)

    @R.function
    def function(x: R.Tensor((2,), "float32")):
        y: (lambda dtype: R.Tensor(tuple(size for size in sizes), dtype))("float32") = R.add(x, x)
        return y

    assert function.ret_ty.shape[0] == 2
    assert function.ret_ty.dtype == "float32"


def test_active_lexical_ancestor_retains_annotation_only_names():
    def outer(extent):
        def middle(width):
            @T.prim_func
            def function(output: T.Buffer((extent, width), "int32")):
                output[0, 0] = VALUE

            return function

        return middle(3)

    for extent in (4, 8):
        function = outer(extent)
        assert [int(value) for value in function.params[0].ty.shape] == [extent, 3]
        assert function.body.value.value == 1


def test_active_lexical_ancestor_respects_nearest_annotation_binding():
    def outer(extent):
        def middle():
            extent = 3

            @T.prim_func
            def function(output: T.Buffer((extent,), "int32")):
                output[0] = VALUE

            return function

        return middle()

    function = outer(8)
    assert [int(value) for value in function.params[0].ty.shape] == [3]


def test_unrelated_same_file_caller_does_not_supply_annotation_locals():
    def build():
        @T.prim_func
        def function(output: T.Buffer((EXTENT,), "int32")):
            output[0] = VALUE

        return function

    def caller():
        EXTENT = 99  # noqa: F841
        VALUE = 99  # noqa: F841
        return build()

    function = caller()
    assert [int(value) for value in function.params[0].ty.shape] == [11]
    assert function.body.value.value == 1


def test_inactive_lexical_ancestor_is_not_replaced_by_unrelated_caller():
    def outer(extent):
        def middle():
            @T.prim_func
            def function(output: T.Buffer((extent,), "int32")):
                output[0] = VALUE

            return function

        return middle

    middle = outer(8)
    extent = 99  # noqa: F841
    with pytest.raises(tvm.error.DiagnosticError, match="extent"):
        middle()


def test_enclosing_class_locals_are_not_nested_annotation_scope():
    class Outer:
        EXTENT = 99

        class Inner:
            @T.prim_func
            def function(output: T.Buffer((EXTENT,), "int32")):
                output[0] = EXTENT

    function = Outer.Inner.function
    assert [int(value) for value in function.params[0].ty.shape] == [11]
    assert function.body.value.value == 11


@pytest.mark.parametrize("factory", [False, True])
def test_jit_retains_active_lexical_ancestor_snapshot(factory):
    decorator = T.jit() if factory else T.jit

    def outer(extent):
        def middle(width):
            @decorator
            def function(output: T.Buffer((extent, width), "int32")):
                output[0, 0] = VALUE

            return function

        pending = middle(3)
        extent = 99
        return pending

    function = outer(8).specialize()
    assert [int(value) for value in function.params[0].ty.shape] == [8, 3]
    assert function.body.value.value == 1


def test_unrelated_intervening_call_ends_annotation_capture():
    def invoke(callback):
        return callback()

    def outer(extent):
        def middle():
            @T.prim_func
            def function(output: T.Buffer((extent,), "int32")):
                output[0] = VALUE

            return function

        return invoke(middle)

    with pytest.raises(tvm.error.DiagnosticError, match="extent"):
        outer(8)
