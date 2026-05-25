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
from tvm.tirx import Buffer


def test_buffer():
    m = tvm.tirx.SizeVar("m", "int32")
    n = tvm.tirx.SizeVar("n", "int32")
    l = tvm.tirx.SizeVar("l", "int32")
    Ab = tvm.tirx.decl_buffer((m, n), "float32")
    Bb = tvm.tirx.decl_buffer((n, l), "float32")

    assert isinstance(Ab, tvm.tirx.Buffer)
    assert Ab.dtype == "float32"
    assert tuple(Ab.shape) == (m, n)


def test_buffer_access_ptr():
    m = tvm.tirx.SizeVar("m", "int32")
    n = tvm.tirx.SizeVar("n", "int32")
    Ab = tvm.tirx.decl_buffer((m, n), "float32", strides=[n + 1, 1])
    aptr = Ab.access_ptr("rw")
    tvm.ir.assert_structural_equal(aptr.args[3], Ab.strides[0] * m)
    assert aptr.args[0].dtype == Ab.dtype
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("w")
    assert aptr.args[4].value == Buffer.WRITE


def test_buffer_access_ptr_offset():
    m = tvm.tirx.SizeVar("m", "int32")
    n = tvm.tirx.SizeVar("n", "int32")
    Ab = tvm.tirx.decl_buffer((m, n), "float32")
    aptr = Ab.access_ptr("rw", offset=100)
    tvm.testing.assert_prim_expr_equal(aptr.args[2], 100)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    v = tvm.tirx.SizeVar("int32", "int32")
    aptr = Ab.access_ptr("rw", offset=100 + 100 + v)
    tvm.testing.assert_prim_expr_equal(aptr.args[2], 200 + v)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("rw", offset=tvm.tirx.call_extern("int32", "test_call", 100 + 100 + v))
    tvm.testing.assert_prim_expr_equal(
        aptr.args[2], tvm.tirx.call_extern("int32", "test_call", 200 + v)
    )
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE


def test_buffer_access_ptr_extent():
    m = tvm.tirx.SizeVar("m", "int32")
    n = tvm.tirx.SizeVar("n", "int32")
    Ab = tvm.tirx.decl_buffer((m, n), "float32")
    aptr = Ab.access_ptr("rw")
    tvm.ir.assert_structural_equal(aptr.args[3], m * n)
    aptr = Ab.access_ptr("rw", offset=100)
    tvm.ir.assert_structural_equal(aptr.args[3], m * n - 100)
    Ab = tvm.tirx.decl_buffer((m, n), "float32", strides=[n + 1, 1])
    aptr = Ab.access_ptr("rw", offset=100)
    tvm.ir.assert_structural_equal(aptr.args[3], Ab.strides[0] * m - 100)

    # Test extent from input params
    aptr = Ab.access_ptr("rw", extent=200)
    tvm.ir.assert_structural_equal(aptr.args[3], T.int32(200))
    aptr = Ab.access_ptr("rw", offset=100, extent=100)
    tvm.ir.assert_structural_equal(aptr.args[3], T.int32(100))


def test_buffer_from_ptr_buffer_load():
    @T.prim_func(private=True, s_tir=True)
    def actual(A: T.Buffer((32,), "float32")):
        B = T.buffer_from_ptr(A[16], shape=(16,))
        B[0] = T.float32(1)

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((32,), "float32")):
        B_data: T.let[T.handle("float32", "global")] = T.address_of(A[16])
        B = T.decl_buffer((16,), "float32", data=B_data)
        B[0] = T.float32(1)

    tvm.ir.assert_structural_equal(actual, expected)


def test_buffer_from_ptr_raw_pointer():
    @T.prim_func(private=True, s_tir=True)
    def actual(A_data: T.handle("float32")):
        A = T.buffer_from_ptr(A_data, shape=(16,))
        A[0] = T.float32(1)

    @T.prim_func(private=True, s_tir=True)
    def expected(A_data: T.handle("float32")):
        A = T.decl_buffer((16,), "float32", data=A_data)
        A[0] = T.float32(1)

    tvm.ir.assert_structural_equal(actual, expected)


def test_buffer_from_ptr_rejects_buffer_without_explicit_pointer():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func(private=True, s_tir=True)
        def _func(A: T.Buffer((16,), "float32")):
            B = T.buffer_from_ptr(A, shape=(16,))
            B[0] = T.float32(1)


def test_tile_op_canonicalizes_buffer_load_to_region():
    @T.prim_func(s_tir=True)
    def func(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        with T.kernel():
            T.copy(B[1], A[2])

    op = func.body.body
    dst = op.args[0]
    src = op.args[1]
    assert isinstance(dst, tvm.tirx.BufferRegion)
    assert isinstance(src, tvm.tirx.BufferRegion)
    tvm.ir.assert_structural_equal(dst.region[0].min, T.int32(1))
    tvm.ir.assert_structural_equal(dst.region[0].extent, T.int32(1))
    tvm.ir.assert_structural_equal(src.region[0].min, T.int32(2))
    tvm.ir.assert_structural_equal(src.region[0].extent, T.int32(1))


def test_tile_op_canonicalizes_optional_buffer_load_operands_to_region():
    @T.prim_func(s_tir=True)
    def func(
        A: T.Buffer((4,), "float32"),
        B: T.Buffer((4,), "float32"),
        C: T.Buffer((4,), "float32"),
    ):
        with T.kernel():
            T.add(C[0], A[1], B[2])

    op = func.body.body
    dst, src1, src2 = op.args
    assert isinstance(dst, tvm.tirx.BufferRegion)
    assert isinstance(src1, tvm.tirx.BufferRegion)
    assert isinstance(src2, tvm.tirx.BufferRegion)
    tvm.ir.assert_structural_equal(dst.region[0].min, T.int32(0))
    tvm.ir.assert_structural_equal(dst.region[0].extent, T.int32(1))
    tvm.ir.assert_structural_equal(src1.region[0].min, T.int32(1))
    tvm.ir.assert_structural_equal(src1.region[0].extent, T.int32(1))
    tvm.ir.assert_structural_equal(src2.region[0].min, T.int32(2))
    tvm.ir.assert_structural_equal(src2.region[0].extent, T.int32(1))


def test_overloaded_tile_op_canonicalizes_buffer_load_operands_to_region():
    @T.prim_func(s_tir=True)
    def unary(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.kernel():
            T.exp(C[0], A[1])

    op = unary.body.body
    dst, src = op.args[:2]
    assert isinstance(dst, tvm.tirx.BufferRegion)
    assert isinstance(src, tvm.tirx.BufferRegion)
    tvm.ir.assert_structural_equal(dst.region[0].min, T.int32(0))
    tvm.ir.assert_structural_equal(dst.region[0].extent, T.int32(1))
    tvm.ir.assert_structural_equal(src.region[0].min, T.int32(1))
    tvm.ir.assert_structural_equal(src.region[0].extent, T.int32(1))

    @T.prim_func(s_tir=True)
    def reduction(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.kernel():
            T.max(C, A[2])

    op = reduction.body.body
    dst, src = op.args[:2]
    assert isinstance(dst, tvm.tirx.BufferRegion)
    assert isinstance(src, tvm.tirx.BufferRegion)
    tvm.ir.assert_structural_equal(dst.region[0].min, T.int32(0))
    tvm.ir.assert_structural_equal(dst.region[0].extent, T.int32(4))
    tvm.ir.assert_structural_equal(src.region[0].min, T.int32(2))
    tvm.ir.assert_structural_equal(src.region[0].extent, T.int32(1))


def test_overloaded_tile_op_with_options_does_not_fall_back_to_expression():
    @T.prim_func(s_tir=True)
    def unary(A: T.Buffer((4,), "float32"), Bias: T.Buffer((4,), "float32")):
        with T.kernel():
            T.exp(A[0], bias=Bias[1])

    op = unary.body.body
    dst, src, bias = op.args[:3]
    assert isinstance(dst, tvm.tirx.BufferRegion)
    assert isinstance(src, tvm.tirx.BufferRegion)
    assert isinstance(bias, tvm.tirx.BufferRegion)
    tvm.ir.assert_structural_equal(dst.region[0].min, T.int32(0))
    tvm.ir.assert_structural_equal(dst.region[0].extent, T.int32(1))
    tvm.ir.assert_structural_equal(src.region[0].min, T.int32(0))
    tvm.ir.assert_structural_equal(src.region[0].extent, T.int32(1))
    tvm.ir.assert_structural_equal(bias.region[0].min, T.int32(1))
    tvm.ir.assert_structural_equal(bias.region[0].extent, T.int32(1))

    @T.prim_func(s_tir=True)
    def reduction(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        with T.kernel():
            T.max(A[0], B[1], axes=0)

    op = reduction.body.body
    dst, src = op.args[:2]
    assert isinstance(dst, tvm.tirx.BufferRegion)
    assert isinstance(src, tvm.tirx.BufferRegion)
    tvm.ir.assert_structural_equal(dst.region[0].min, T.int32(0))
    tvm.ir.assert_structural_equal(dst.region[0].extent, T.int32(1))
    tvm.ir.assert_structural_equal(src.region[0].min, T.int32(1))
    tvm.ir.assert_structural_equal(src.region[0].extent, T.int32(1))


def test_ambiguous_buffer_load_max_remains_expression():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        B[0] = T.max(A[1], B[2])

    assert isinstance(func.body.value, tvm.tirx.PrimExpr)


def test_expression_dtype_kwarg_does_not_force_tile_op():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        B[0] = T.exp(A[1], dtype="float32") + T.max(A[1], B[2], dtype="float32")

    assert isinstance(func.body.value, tvm.tirx.PrimExpr)


def test_dynamic_tile_op_canonicalizes_buffer_load_to_region():
    @T.prim_func(s_tir=True)
    def func(A: T.Buffer((4,), "float32")):
        with T.kernel():
            T.test_buffer_load_region(A[2])

    arg = func.body.body.args[0]
    assert isinstance(arg, tvm.tirx.BufferRegion)
    tvm.ir.assert_structural_equal(arg.region[0].min, T.int32(2))
    tvm.ir.assert_structural_equal(arg.region[0].extent, T.int32(1))


def test_buffer_load_to_region_canonicalization_handles_ramp():
    @T.prim_func(s_tir=True)
    def func(A: T.Buffer((16,), "float32")):
        with T.sblock("read"):
            T.reads(A[T.Ramp(4, 1, 4)])
            T.evaluate(0)

    read_region = func.body.block.reads[0]
    tvm.ir.assert_structural_equal(read_region.region[0].min, T.int32(4))
    tvm.ir.assert_structural_equal(read_region.region[0].extent, T.int32(4))


def test_buffer_vload():
    m = tvm.tirx.SizeVar("m", "int32")
    n = tvm.tirx.SizeVar("n", "int32")
    Ab = tvm.tirx.decl_buffer((m, n), "float32", elem_offset=100)
    load = Ab.vload([2, 3])
    tvm.ir.assert_structural_equal(load.indices, [T.int32(2), T.int32(3)])


def test_buffer_offset_of():
    m = tvm.tirx.SizeVar("m", "int32")
    n = tvm.tirx.SizeVar("n", "int32")
    Ab = tvm.tirx.decl_buffer((m, n), "float32", elem_offset=100)
    offset = Ab.offset_of([2, 3])
    tvm.ir.assert_structural_equal(offset, [n * 2 + 103])


def test_buffer_index_merge_mult_mod():
    m = tvm.tirx.SizeVar("m", "int32")
    n = tvm.tirx.SizeVar("n", "int32")
    s = tvm.tirx.SizeVar("s", "int32")
    k0 = tvm.tirx.SizeVar("k0", "int32")
    k1 = tvm.tirx.SizeVar("k1", "int32")
    A = tvm.tirx.decl_buffer((m, n), "float32")
    A_stride = tvm.tirx.decl_buffer((m, n), "float32", strides=(s, 1))

    def assert_simplified_equal(index_simplified, index_direct):
        (
            tvm.ir.assert_structural_equal(index_simplified, index_direct),
            f"index_simplified={index_simplified}, index_direct={index_direct}",
        )

    idxd = tvm.tirx.indexdiv
    idxm = tvm.tirx.indexmod

    # Test Case1
    index_simplified = A_stride.offset_of(
        (idxd(idxm(k0, k1), s), idxm(idxm(k0, k1), s) + idxd(k0, k1) * k1)
    )
    index_direct = A_stride.offset_of((0, k0))
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case2
    index_simplified = A.offset_of(
        (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, s)), n) + idxm(k0, k1))
    )
    index_direct = A.offset_of((0, idxm(k0, idxd(k1, s)) + idxm(k0, k1)))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case3
    index_simplified = A.offset_of(
        (
            idxd((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxd(idxm(k0, idxd(k1, s)), n),
            idxm((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxm(idxm(k0, idxd(k1, s)), n),
        )
    )
    index_direct = A.offset_of((0, k0))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case4 (not able to simplify)
    index_simplified = A.offset_of(
        (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1))
    )
    index_direct = A.offset_of(
        (0, idxd(idxm(k0, idxd(k1, s)), n) * n + (idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1)))
    )
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case5
    B = tvm.tirx.decl_buffer((1, 14, 14, 1024))
    i = tvm.tirx.SizeVar("i", "int32")
    j = tvm.tirx.SizeVar("j", "int32")
    k = tvm.tirx.SizeVar("k", "int32")

    index_simplified1 = B.offset_of(
        (
            idxd(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd((i * 50176 + j * 28672 + k), 1024), 14),
            idxm((i * 50176 + j * 28672 + k), 1024),
        )
    )
    index_simplified2 = B.offset_of(
        (
            idxd(idxd(i * 49 + j * 28 + idxd(k, 1024), 14), 14),
            idxm(idxd(i * 49 + j * 28 + idxd(k, 1024), 14), 14),
            idxm(i * 7 + idxd(k, 1024), 14),
            idxm(k, 1024),
        )
    )
    index_direct = B.offset_of((0, 0, 0, (i * 50176 + j * 28672 + k)))
    assert_simplified_equal(index_simplified1, index_direct)
    assert_simplified_equal(index_simplified2, index_direct)


def test_buffer_flatten():
    """A buffer should flatten to a 1-d shape"""
    buf = tvm.tirx.decl_buffer([16, 32])
    flat = buf.get_flattened_buffer()
    assert buf.data.same_as(flat.data)
    tvm.ir.assert_structural_equal(flat.shape, [T.int32(16 * 32)])


def test_buffer_flatten_preserves_identity():
    """Flattening a 1-d buffer should return the original"""
    buf = tvm.tirx.decl_buffer([16])
    flat = buf.get_flattened_buffer()
    assert buf.same_as(flat)


def test_buffer_flatten_uses_axis_separators():
    """Flattening to N-d physical buffers uses the axis separators"""
    buf = tvm.tirx.decl_buffer([4, 16, 32], axis_separators=[2])
    flat = buf.get_flattened_buffer()
    tvm.ir.assert_structural_equal(flat.axis_separators, [T.int32(1)])
    tvm.ir.assert_structural_equal(flat.shape, [T.int32(4 * 16), T.int32(32)])


def test_invalid_axis_separators_raises_exception():
    with pytest.raises(ValueError):
        tvm.tirx.decl_buffer([1], axis_separators=[1, 2])


if __name__ == "__main__":
    tvm.testing.main()
