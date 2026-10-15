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

import uuid

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T


def test_tensor_pointer_type_derived_from_dtype_and_scope():
    data = tvm_tirx.ir.Var("storage", tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("uint8"), "local"))
    tensor = tvm_tirx.tirx.decl_tensor((8,), "float16", data=data)

    assert tensor.ty.dtype == tvm_tirx.ir.PrimType("float16")
    assert tensor.ty.scope == tvm_tirx.runtime.MemoryScope("local")
    assert tensor.data.ty == tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float16"), "local")


@pytest.mark.parametrize("scope", ["global", "shared", tvm_tirx.runtime.MemoryScope("local")])
def test_tensor_type_constructor_scope_order(scope):
    direct = tvm_tirx.tirx.TensorType((4, 8), "float32", scope, (16, 1), 4, 64, 2)
    scripted = T.Tensor((4, 8), "float32", scope, (16, 1), 4, 64, 2)
    declared = tvm_tirx.tirx.decl_tensor(
        (4, 8),
        "float32",
        scope=scope,
        strides=(16, 1),
        elem_offset=4,
        data_alignment=64,
        offset_factor=2,
    )
    tvm_tirx.ir.assert_structural_equal(direct, scripted)
    tvm_tirx.ir.assert_structural_equal(direct, declared.ty)
    assert direct.scope.same_as(tvm_tirx.runtime.MemoryScope(scope))


@pytest.mark.parametrize("scope", ["", "tensor_type.unregistered"])
def test_tensor_type_rejects_unregistered_scope(scope):
    with pytest.raises(ValueError):
        tvm_tirx.tirx.TensorType((4,), "float32", scope)
    with pytest.raises(ValueError):
        T.Tensor((4,), "float32", scope)
    with pytest.raises(ValueError):
        tvm_tirx.tirx.decl_tensor((4,), "float32", scope=scope)


def test_decl_tensor_physical_data_binding():
    tensor = tvm_tirx.tirx.decl_tensor((8,), "float32")
    data = tvm_tirx.tirx.Var("data", tensor.data.ty)

    decl = tvm_tirx.ir.Bind(
        tensor,
        tvm_tirx.ir.Call(
            "tirx.decl_tensor",
            [data],
            ty_args=[tensor.ty],
        ),
    )
    assert decl.var.same_as(tensor)
    assert decl.value.args[0].same_as(data)


def test_tensor_ptr_to():
    m = tvm_tirx.tirx.Var("m", "int32")
    n = tvm_tirx.tirx.Var("n", "int32")
    tensor = tvm_tirx.tirx.decl_tensor((m, n), "float32", strides=[n + 1, 1], elem_offset=7)
    pointer = tensor.ptr_to([2, 3])
    assert pointer.op.name == "tirx.address_of"
    assert pointer.ty == tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"))
    assert pointer.args[0].source.same_as(tensor)
    tvm_tirx.ir.assert_structural_equal(pointer.args[0].indices, [T.int32(2), T.int32(3)])
    shared = tvm_tirx.tirx.decl_tensor((m, n), "float32", scope="shared")
    assert shared.ptr_to([0, 0]).ty == tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"), "shared")


@pytest.mark.parametrize("dtype,scope", [("float32", "global"), ("bool", "shared")])
def test_tensor_type_and_projection(dtype, scope):
    m, n = tvm_tirx.ir.Var("m", tvm_tirx.ir.PrimType("int32")), tvm_tirx.ir.Var("n", tvm_tirx.ir.PrimType("int32"))
    tensor = tvm_tirx.tirx.decl_tensor((m, n), dtype, scope=scope)
    assert type(tensor) is tvm_tirx.ir.Var
    assert tvm_tirx.tirx.is_tensor_var(tensor)
    assert not tvm_tirx.tirx.is_tensor_var(m)
    assert isinstance(tensor.ty, tvm_tirx.tirx.TensorType)
    assert tuple(tensor.ty.shape) == (m, n)
    assert tuple(tensor.shape) == (m, n)
    assert tensor.ty.dtype == tvm_tirx.ir.PrimType(dtype)
    assert tensor.dtype == tvm_tirx.DataType(dtype)
    assert tensor.data.ty == tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType(dtype), scope)
    assert tensor.data.op.name == "tirx.tensor_data_ptr"
    assert tensor.data.args[0].same_as(tensor)
    serialized = tvm_tirx.ir.save_json(tensor.ty)
    assert '"tirx.TensorType"' in serialized
    restored = tvm_tirx.ir.load_json(serialized)
    assert isinstance(restored, tvm_tirx.tirx.TensorType)
    tvm_tirx.ir.assert_structural_equal(restored, tensor.ty, map_free_vars=True)
    for name in ("shape", "dtype", "data"):
        assert not hasattr(m, name)
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(m, name)


def test_tensor_type_default_offset():
    tensor = tvm_tirx.tirx.decl_tensor((8,), "float32", elem_offset=None)
    assert int(tensor.ty.elem_offset) == 0


@pytest.mark.parametrize(
    "dtype",
    [
        "float4_e2m1fn",
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ],
)
def test_tensor_type_low_precision_dtype(dtype):
    m = tvm_tirx.ir.Var("m", tvm_tirx.ir.PrimType("int32"))
    n = tvm_tirx.ir.Var("n", tvm_tirx.ir.PrimType("int32"))
    tensor = tvm_tirx.tirx.decl_tensor((m, n), dtype)
    assert tensor.dtype == dtype


def test_memory_scope_typed_conversion():
    custom = MemoryScope.register("test.tensor.scope." + uuid.uuid4().hex)
    for scope in (custom, custom.name):
        ptr_ty = tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"), scope)
        tensor = tvm_tirx.tirx.decl_tensor((4,), "float32", scope=scope)
        tensor_ty = tvm_tirx.tirx.TensorType((4,), "float32", scope)
        assert tensor_ty.scope.same_as(custom)
        assert ptr_ty.scope.same_as(custom)
        assert tensor.ty.scope.same_as(custom)
        assert tensor.scope().same_as(custom)
        assert tensor.data.ty.scope.same_as(custom)

    unknown = "test.memory_scope.unknown." + uuid.uuid4().hex
    for construct in (
        lambda: tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"), unknown),
        lambda: tvm_tirx.tirx.decl_tensor((4,), "float32", scope=unknown),
        lambda: T.Tensor((4,), "float32", scope=unknown),
    ):
        with pytest.raises(ValueError):
            construct()
    with pytest.raises(ValueError):
        MemoryScope(unknown)


def test_memory_scope_global_defaults():
    global_scope = MemoryScope("global")
    assert tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32")).scope.same_as(global_scope)
    assert tvm_tirx.tirx.decl_tensor((4,), "float32").scope().same_as(global_scope)
    for scope in ("global", global_scope):
        assert tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"), scope).scope.same_as(global_scope)
        assert tvm_tirx.tirx.decl_tensor((4,), "float32", scope=scope).scope().same_as(global_scope)
        assert T.Tensor((4,), "float32", scope=scope).scope.same_as(global_scope)
    for construct in (
        lambda: MemoryScope(""),
        lambda: tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"), ""),
        lambda: tvm_tirx.tirx.decl_tensor((4,), "float32", scope=""),
        lambda: T.Tensor((4,), "float32", scope=""),
    ):
        with pytest.raises(ValueError):
            construct()


def test_tensor_type_memory_scope_json_roundtrip():
    custom = MemoryScope.register("test.ir.scope." + uuid.uuid4().hex)
    for scope in (MemoryScope("global"), MemoryScope("shared.dyn"), custom):
        ty = tvm_tirx.tirx.TensorType((4,), "float32", scope=scope)
        restored = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(ty))
        assert restored.scope.same_as(scope)
        tvm_tirx.ir.assert_structural_equal(ty, restored)


if __name__ == "__main__":
    tvm_tirx.testing.main()
