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
# ruff: noqa: E741, F401, F841

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T


def test_buffer():
    m = tvm.tirx.Var("m", "int32")
    n = tvm.tirx.Var("n", "int32")
    l = tvm.tirx.Var("l", "int32")
    Ab = tvm.tirx.decl_tensor((m, n), "float32")
    Bb = tvm.tirx.decl_tensor((n, l), "float32")

    assert type(Ab) is tvm.ir.Var
    assert tvm.tirx.is_tensor_var(Ab)
    assert isinstance(Ab.ty, tvm.tirx.TensorType)
    assert Ab.ty.dtype == tvm.ir.PrimType("float32")
    assert tuple(Ab.ty.shape) == (m, n)
    assert not tvm.tirx.is_tensor_var(m)

    serialized = tvm.ir.save_json(Ab.ty)
    assert '"tirx.TensorType"' in serialized
    restored = tvm.ir.load_json(serialized)
    assert isinstance(restored, tvm.tirx.TensorType)
    tvm.ir.assert_structural_equal(restored, Ab.ty, map_free_vars=True)


def test_tensor_var_identity_and_global_var_properties():
    scalar = tvm.ir.Var("scalar", tvm.ir.PrimType("int32"))
    buffer = tvm.tirx.decl_tensor((8,), "float32")

    assert isinstance(buffer, tvm.ir.Var)
    assert isinstance(scalar, tvm.ir.Var)
    assert not tvm.tirx.is_tensor_var(scalar)
    assert tvm.tirx.is_tensor_var(buffer)

    assert tuple(buffer.shape) == (8,)
    assert buffer.dtype == tvm.DataType("float32")
    assert buffer.data.args[0].same_as(buffer)

    for name in ("shape", "dtype", "data"):
        assert not hasattr(scalar, name)
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(scalar, name)


def test_buffer_data_is_typed_projection():
    buffer = tvm.tirx.decl_tensor((8,), "bool", scope="shared")

    assert buffer.ty.dtype == tvm.ir.PrimType("bool")
    assert buffer.data.ty == tvm.ir.PtrType(tvm.ir.PrimType("bool"), "shared")
    assert buffer.data.op.name == "tirx.tensor_data_ptr"
    assert buffer.data.args[0].same_as(buffer)


def test_buffer_pointer_type_derived_from_dtype_and_scope():
    data = tvm.ir.Var("storage", tvm.ir.PtrType(tvm.ir.PrimType("uint8"), "local"))
    buffer = tvm.tirx.decl_tensor((8,), "float16", data=data)

    assert buffer.ty.dtype == tvm.ir.PrimType("float16")
    assert buffer.ty.storage_scope == "local"
    assert buffer.data.ty == tvm.ir.PtrType(tvm.ir.PrimType("float16"), "local")


def test_decl_buffer_physical_data_binding():
    buffer = tvm.tirx.decl_tensor((8,), "float32")
    data = tvm.tirx.Var("data", buffer.data.ty)

    decl = tvm.ir.Bind(
        buffer,
        tvm.ir.Call(
            "tirx.decl_tensor",
            [
                data,
                tvm.ir.Tuple(buffer.shape),
                tvm.ir.DataTypeImm(tvm.DataType(buffer.dtype)),
                tvm.ir.StringImm(buffer.scope()),
            ],
            ty=buffer.ty,
        ),
    )
    assert decl.var.same_as(buffer)
    assert decl.value.args[0].same_as(data)


def test_buffer_ptr_to():
    m = tvm.tirx.Var("m", "int32")
    n = tvm.tirx.Var("n", "int32")
    buffer = tvm.tirx.decl_tensor((m, n), "float32", strides=[n + 1, 1], elem_offset=7)
    pointer = buffer.ptr_to([2, 3])
    assert pointer.op.name == "tirx.address_of"
    assert pointer.ty == tvm.ir.PtrType(tvm.ir.PrimType("float32"))
    assert pointer.args[0].source.same_as(buffer)
    tvm.ir.assert_structural_equal(pointer.args[0].indices, [T.int32(2), T.int32(3)])
    shared = tvm.tirx.decl_tensor((m, n), "float32", scope="shared")
    assert shared.ptr_to([0, 0]).ty == tvm.ir.PtrType(tvm.ir.PrimType("float32"), "shared")


def _flattened_load(buffer, indices):
    function = tvm.tirx.Function([buffer], [tvm.ir.Evaluate(buffer[tuple(indices)])])
    module = tvm.tirx.transform.FlattenBuffer()(tvm.IRModule({"main": function}))
    return module["main"].body[-1].value


def _flattened_indices(buffer, indices):
    return _flattened_load(buffer, indices).indices


def test_buffer_load():
    m = tvm.tirx.Var("m", "int32")
    n = tvm.tirx.Var("n", "int32")
    Ab = tvm.tirx.decl_tensor((m, n), "float32", elem_offset=100)
    load = Ab[2, 3]
    tvm.ir.assert_structural_equal(load.indices, [T.int32(2), T.int32(3)])


def test_buffer_flatten_offset():
    m = tvm.tirx.Var("m", "int32")
    n = tvm.tirx.Var("n", "int32")
    Ab = tvm.tirx.decl_tensor((m, n), "float32", elem_offset=100)
    offset = _flattened_indices(Ab, [2, 3])
    tvm.ir.assert_structural_equal(offset, [n * 2 + 103])


def test_buffer_index_merge_mult_mod():
    m = tvm.tirx.Var("m", "int32")
    n = tvm.tirx.Var("n", "int32")
    s = tvm.tirx.Var("s", "int32")
    k0 = tvm.tirx.Var("k0", "int32")
    k1 = tvm.tirx.Var("k1", "int32")
    A = tvm.tirx.decl_tensor((m, n), "float32")
    A_stride = tvm.tirx.decl_tensor((m, n), "float32", strides=(s, 1))

    def assert_simplified_equal(index_simplified, index_direct):
        (
            tvm.ir.assert_structural_equal(index_simplified, index_direct),
            f"index_simplified={index_simplified}, index_direct={index_direct}",
        )

    idxd = tvm.tirx.indexdiv
    idxm = tvm.tirx.indexmod

    # Test Case1
    index_simplified = _flattened_indices(
        A_stride, (idxd(idxm(k0, k1), s), idxm(idxm(k0, k1), s) + idxd(k0, k1) * k1)
    )
    index_direct = _flattened_indices(A_stride, (0, k0))
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case2
    index_simplified = _flattened_indices(
        A, (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, s)), n) + idxm(k0, k1))
    )
    index_direct = _flattened_indices(A, (0, idxm(k0, idxd(k1, s)) + idxm(k0, k1)))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case3
    index_simplified = _flattened_indices(
        A,
        (
            idxd((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxd(idxm(k0, idxd(k1, s)), n),
            idxm((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxm(idxm(k0, idxd(k1, s)), n),
        ),
    )
    index_direct = _flattened_indices(A, (0, k0))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case4 (not able to simplify)
    index_simplified = _flattened_indices(
        A, (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1))
    )
    index_direct = _flattened_indices(
        A, (0, idxd(idxm(k0, idxd(k1, s)), n) * n + (idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1)))
    )
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case5
    B = tvm.tirx.decl_tensor((1, 14, 14, 1024))
    i = tvm.tirx.Var("i", "int32")
    j = tvm.tirx.Var("j", "int32")
    k = tvm.tirx.Var("k", "int32")

    index_simplified1 = _flattened_indices(
        B,
        (
            idxd(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd((i * 50176 + j * 28672 + k), 1024), 14),
            idxm((i * 50176 + j * 28672 + k), 1024),
        ),
    )
    index_simplified2 = _flattened_indices(
        B,
        (
            idxd(idxd(i * 49 + j * 28 + idxd(k, 1024), 14), 14),
            idxm(idxd(i * 49 + j * 28 + idxd(k, 1024), 14), 14),
            idxm(i * 7 + idxd(k, 1024), 14),
            idxm(k, 1024),
        ),
    )
    index_direct = _flattened_indices(B, (0, 0, 0, (i * 50176 + j * 28672 + k)))
    assert_simplified_equal(index_simplified1, index_direct)
    assert_simplified_equal(index_simplified2, index_direct)


@pytest.mark.parametrize(
    "shape,strides,expected",
    [([16, 32], None, 512), ([16], None, 16), ([], None, 1), ([16, 32], [40, 1], 640)],
)
def test_buffer_flatten(shape, strides, expected):
    """Flattening derives the storage span and binds a fresh physical view."""
    buf = tvm.tirx.decl_tensor(shape, strides=strides)
    flat = _flattened_load(buf, [0] * len(shape)).source
    assert not buf.same_as(flat)
    assert flat.data.args[0].same_as(flat)
    assert flat.data.op.name == "tirx.tensor_data_ptr"
    tvm.ir.assert_structural_equal(flat.ty.shape, [T.int32(expected)])


if __name__ == "__main__":
    tvm.testing.main()
