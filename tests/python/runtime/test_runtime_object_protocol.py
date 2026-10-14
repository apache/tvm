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
"""Native FFI object conversion for TVM wrappers and script helpers."""

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import ir, relax, te, tirx
from tvm.relax.script.ir_builder import op as relax_op
from tvm.script.ir_builder.stmt import _as_expr
from tvm.tirx.script.ir_builder import op as tir_op
from tvm.tirx.tile.instruction import _expr


class ObjectWrapper:
    def __init__(self, value):
        self.value = value

    def __tvm_ffi_object__(self) -> tvm_ffi.Object:
        return self.value


def test_object_protocol_identity():
    value = ir.Var("x", "int32")
    wrapped = ObjectWrapper(value)
    assert tvm_ffi.convert(wrapped) is value
    echo = tvm_ffi.convert_func(lambda x: x)
    assert echo(wrapped).same_as(value)
    assert tvm_ffi.convert([wrapped])[0].same_as(value)
    assert tvm_ffi.convert({"value": [wrapped]})["value"][0].same_as(value)
    assert _as_expr(wrapped) is value
    assert _expr(wrapped) is value


@pytest.mark.parametrize("unequal", [False, True])
def test_deferred_comparison_protocol(unequal):
    x, y = ir.Var("x", "int32"), ir.Var("y", "int32")
    comparison = x != y if unequal else x == y
    assert bool(comparison) is unequal
    assert bool(x != x if unequal else x == x) is not unequal
    result = tvm_ffi.convert(comparison)
    assert isinstance(result, tirx.NE if unequal else tirx.EQ)
    assert result.a.same_as(x)
    assert result.b.same_as(y)
    echo = tvm_ffi.convert_func(lambda x: x)
    tvm.ir.assert_structural_equal(echo(comparison), result)
    tvm.ir.assert_structural_equal(tvm_ffi.convert([comparison])[0], result)
    assert isinstance(tir_op.logical_not(comparison), tirx.Not)
    assert isinstance(tir_op.logical_and(comparison, True), tirx.And)
    assert isinstance(tir_op.logical_or(comparison, False), tirx.Or)
    tvm.ir.assert_structural_equal(_as_expr(comparison), result)
    tvm.ir.assert_structural_equal(_expr(comparison), result)


def test_tensor_slice_protocol():
    tensor = te.placeholder((2, 3), name="A")
    sliced = tensor[0][1]
    result = tvm_ffi.convert(sliced)
    assert result.op.same_as(ir.Op.get("te.tensor_load"))
    assert result.args[0].same_as(tensor)
    assert list(result.args[1:]) == [0, 1]
    echo = tvm_ffi.convert_func(lambda x: x)
    tvm.ir.assert_structural_equal(echo(sliced), result)
    tvm.ir.assert_structural_equal(tvm_ffi.convert([sliced])[0], result)


def test_enum_protocol():
    enum = tirx.CallEffectKind.Pure
    result = tvm_ffi.convert(enum)
    assert isinstance(result, tirx.IntImm)
    assert result.value == 1
    assert result.ty == ir.PrimType("int32")
    echo = tvm_ffi.convert_func(lambda x: x)
    tvm.ir.assert_structural_equal(echo(enum), result)
    tvm.ir.assert_structural_equal(tvm_ffi.convert([enum])[0], result)


def test_builder_protocol_preserves_host_values():
    host = object()
    assert tir_op.select(True, host, None) is host
    assert relax_op.if_then_else_(True, host, None) is host
    assert tir_op.logical_and(True, host) is host
    assert tir_op.logical_or(False, host) is host
    x = ir.Var("x", "int32")
    condition = ObjectWrapper(tvm_ffi.convert(x == 0))
    true_value = ObjectWrapper(x)
    tvm.ir.assert_structural_equal(
        tir_op.select(condition, true_value, 0), tir_op.select(condition.value, x, 0)
    )
    tvm.ir.assert_structural_equal(
        relax_op.if_then_else_(condition, true_value, 0),
        relax_op.if_then_else_(condition.value, x, 0),
    )


@pytest.mark.parametrize(
    "make_call", [relax.op.call_builtin_with_ctx, relax.op.call_pure_packed, relax_op.call_packed]
)
def test_wrapped_type_arguments(make_call):
    ty = ir.PrimType("int32")
    call = make_call("custom", (), ty_args=[ObjectWrapper(ty)])
    assert call.ty_args[0].same_as(ty)
    call = make_call("custom", (), ty_args=[lambda: ty])
    assert call.ty_args[0].same_as(ty)


def test_invalid_object_protocol_result():
    echo = tvm_ffi.convert_func(lambda x: x)
    with pytest.raises(TypeError):
        echo(ObjectWrapper(1))
    with pytest.raises(TypeError):
        tvm_ffi.convert([ObjectWrapper(1)])


if __name__ == "__main__":
    tvm.testing.main()
