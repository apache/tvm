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
"""Public construction preserves validation, source locations, and lexical bindings."""

import numpy as np
import pytest

import tvm
from tvm import ir, relax, tirx
from tvm.error import DiagnosticError
from tvm.relax import BasePyModule
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.script.parser.source import Source

VALUE = 1
MACRO_VALUE = 2


def _build_lexical_global():
    @T.prim_func
    def write(A: T.Buffer((1,), "int32")):
        A[0] = VALUE

    return write


def _build_symbolic_functions():
    @I.ir_module
    class Module:
        @R.function
        def first(x: R.Tensor(("n",), "float32")):
            n = T.int64()  # noqa: F841
            return x

        @R.function
        def second(x: R.Tensor(("n",), "float32")):
            n = T.int64()  # noqa: F841
            return x

    return Module


def test_ordinary_class_member_constructs_prim_func():
    class Holder:
        @T.prim_func(s_tir=True)
        def function():
            T.evaluate(7)

    assert isinstance(Holder.function, tirx.PrimFunc)
    assert isinstance(Holder.function.body, tirx.Evaluate)
    assert Holder.function.body.value.value == 7


def test_ir_module_defers_members_until_forward_signatures_exist():
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2,), "float32")):
            return Module.identity(x)

        @R.function
        def identity(x: R.Tensor((2,), "float32")):
            return x

    assert isinstance(Module, ir.IRModule)
    call = Module["main"].body.blocks[0].bindings[0].value
    assert isinstance(call, ir.Call)
    assert call.op.same_as(Module.get_global_var("identity"))
    assert call.args[0].same_as(Module["main"].params[0])
    assert Module["identity"].body.body.same_as(Module["identity"].params[0])


def _invalid_inplace_module(**options):
    @I.ir_module(**options)
    class Module:
        @T.prim_func(s_tir=True)
        def kernel(A: T.Buffer((2,), "int32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2,), "int32")):
            R.func_attr({"relax.force_pure": True})
            result = R.call_tir_inplace(
                Module.kernel,
                (x,),
                [0, 0],
                [R.Tensor((2,), "int32"), R.Tensor((2,), "int32")],
            )
            return result

    return Module


def test_default_relax_validation_rejects_repeated_inplace_input():
    with pytest.raises(DiagnosticError):
        _invalid_inplace_module()


def test_relax_validation_can_be_explicitly_disabled():
    module = _invalid_inplace_module(check_well_formed=False)
    assert isinstance(module["main"], relax.Function)
    call = module["main"].body.blocks[0].bindings[0].value
    assert list(call.attrs.inplace_indices) == [0, 0]
    assert not relax.analysis.check_well_formed(module)


def test_default_tirx_validation_rejects_non_divisible_scope_extents():
    with pytest.raises(DiagnosticError):

        @T.prim_func
        def invalid():
            T.device_entry()
            block = T.cta_id([1])
            warp = T.warp_id([3])
            thread = T.thread_id([100])
            T.evaluate(block + warp + thread)


def test_tirx_validation_can_be_explicitly_disabled():
    @T.prim_func(check_well_formed=False)
    def invalid():
        T.device_entry()
        block = T.cta_id([1])
        warp = T.warp_id([3])
        thread = T.thread_id([100])
        T.evaluate(block + warp + thread)

    assert isinstance(invalid, tirx.PrimFunc)
    definitions = invalid.body.body.seq[:3]
    assert all(isinstance(statement, tirx.ScopeIdDefStmt) for statement in definitions)
    with pytest.raises(tvm.error.InternalError, match="100 is not divisible by 3"):
        tirx.analysis.verify_tirx_well_formed(invalid)


def test_python_module_factory_and_attached_function_execute():
    @I.ir_module
    class Module(BasePyModule):
        @I.pyfunc
        def twice(value):
            return value * 2

    assert callable(Module)
    assert Module.pyfuncs["twice"](3) == 6
    instance = Module(tvm.cpu())
    assert isinstance(instance, BasePyModule)
    assert instance.twice(4) == 8
    assert instance.pyfuncs["twice"](5) == 10


def _position(span):
    return span.line, span.column, span.end_line, span.end_column


@pytest.mark.parametrize("entrypoint", ["source", "function"])
def test_source_and_ir_call_spans_use_one_based_columns(entrypoint):
    def direct():
        T.evaluate(T.call_extern("int32", "direct"))

    if entrypoint == "source":
        program = (
            "@T.prim_func(s_tir=True)\ndef direct():\n"
            '    T.evaluate(T.call_extern("int32", "direct"))\n'
        )
        source = Source(program)
        function = tvm.script.from_source(program)
    else:
        source = Source(direct)
        function = T.prim_func(direct, s_tir=True)
    call = source.as_ast().body[0].body[0].value.args[0]
    expected = source.to_span(call)
    assert expected.column == call.col_offset + source.start_column + 1
    assert expected.end_column == call.end_col_offset + source.start_column + 1
    assert _position(function.body.value.span) == _position(expected)
    assert function.body.value.span.source_name.name == source.source_name


def test_nested_helper_spans_preserve_caller_and_definition_columns():
    text = """@T.inline
def inner():
    T.evaluate(T.call_extern("int32", "nested"))
@T.inline
def outer():
    inner()
@T.prim_func(s_tir=True)
def main():
    outer()
"""
    source = Source(text)
    inner, outer, main = source.as_ast().body
    calls = [main.body[0].value, outer.body[0].value, inner.body[0].value.args[0]]
    expected = [_position(source.to_span(call)) for call in calls]
    assert [position[1] for position in expected] == [5, 5, 16]
    function = tvm.script.from_source(text)
    span = function.body.value.span
    assert isinstance(span, ir.SequentialSpan)
    assert [_position(item) for item in span.spans] == expected
    assert all(item.source_name.name == "<str>" for item in span.spans)


def test_optional_annotation_is_restricted_to_jit():
    with pytest.raises(DiagnosticError, match="only supported by @T.jit"):

        @T.prim_func(private=True)
        def invalid(value: T.Optional(T.handle)):
            T.evaluate(0)

    @T.jit(private=True)
    def valid(value: T.Optional(T.handle)):
        if I.constexpr(value is None):
            T.evaluate(1)
        else:
            T.evaluate(2)

    present, absent = valid.specialize(), valid.specialize(value=None)
    assert len(present.params) == 1
    assert len(absent.params) == 0
    assert present.body.value.value == 2
    assert absent.body.value.value == 1


def test_lexical_global_is_not_replaced_by_dynamic_caller_local():
    VALUE = 2  # noqa: F841
    function = _build_lexical_global()
    assert function.body.value.value == 1
    compiled = tvm.compile(function, target="llvm", tir_pipeline="tirx")
    output = tvm.runtime.tensor(np.zeros(1, dtype="int32"))
    compiled(output)
    np.testing.assert_array_equal(output.numpy(), np.array([1], dtype="int32"))


def test_annotation_only_enclosing_binding_is_captured():
    def build():
        shape = (3,)

        @T.prim_func
        def function(A: "T.Buffer(shape, 'int32')"):
            A[0] = 1

        return function

    shape = (7,)  # noqa: F841
    function = build()
    assert [int(extent) for extent in function.params[0].ty.shape] == [3]


def test_dynamic_caller_symbol_does_not_join_function_declarations():
    n = tirx.Var("n", "int64")
    module = _build_symbolic_functions()
    first = module["first"].params[0].ty.shape.ty.values[0]
    second = module["second"].params[0].ty.shape.ty.values[0]
    assert not first.same_as(second)
    assert not first.same_as(n)
    assert not second.same_as(n)


@pytest.mark.parametrize("hygienic,expected", [(True, 2), (False, 1)])
def test_macro_capture_policy_remains_explicit(hygienic, expected, monkeypatch):
    monkeypatch.setitem(globals(), "MACRO_VALUE", 2)

    @T.macro(hygienic=hygienic)
    def write(output):
        output[0] = MACRO_VALUE

    monkeypatch.setitem(globals(), "MACRO_VALUE", 1)

    @T.prim_func
    def function(output: T.Buffer((1,), "int32")):
        write(output)

    assert function.body.value.value == expected


def _define_delayed_annotation(shape):
    @T.jit
    def function(output: "T.Buffer(shape, 'int32')"):
        output[0] = VALUE

    return function


def test_delayed_jit_retains_annotation_only_definition_binding():
    pending = _define_delayed_annotation((3,))
    shape = (7,)  # noqa: F841
    VALUE = 2  # noqa: F841
    function = pending.specialize()
    assert [int(extent) for extent in function.params[0].ty.shape] == [3]
    assert function.body.value.value == 1
