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
"""S-TIR script dynamic shape."""

from __future__ import annotations

import pytest

import tvm.testing
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm.script import from_source
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_tir_starred_shape_expression():
    dims = (128, 128)

    @Ts.prim_func(private=True)
    def starred(A: T.Buffer([128, *dims], "int32")) -> None:
        for i, j, k in T.grid(*A.shape):
            A[i, j, k] = T.int32(1)

    @Ts.prim_func(private=True)
    def non_starred(A: T.Buffer([128, 128, 128], "int32")) -> None:
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_inferred_ty_with_dynamic_buffer():
    """The inferred Type may contain dynamic shapes"""

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @Ts.prim_func
    def func(A: T.Buffer([M, N], "float32"), B: T.Buffer([M * N], "float32")):
        for i, j in T.grid(M, N):
            B[i * N + j] = A[i, j]

    M = tvm.tirx.Var("M", "int64")
    N = tvm.tirx.Var("N", "int64")
    expected = tvm.relax.FuncType(
        [
            tvm.relax.TensorType([M, N], "float32"),
            tvm.relax.TensorType([M * N], "float32"),
        ],
        tvm.relax.TupleType([]),
        purity=False,
    )
    tvm.ir.assert_structural_equal(func.ty, expected)


def test_tir_buffer_region_extent_correct_dtype():
    @Ts.prim_func
    def func(A: T.Buffer((T.int64(16), T.int64(1)), "float32")):
        for i in T.grid(T.int64(16)):
            with Ts.sblock("block"):
                vi = Ts.axis.remap("S", [i])
                Ts.reads(A[vi, T.int64(0) : T.int64(1)])
                T.evaluate(0)

    assert func.body.block.body.body.block.reads[0].region[0].extent.ty.dtype == "int64"


N = T.dynamic("N", "int32")


M = T.dynamic("M", "int32")


K = T.dynamic("K", "int32")


@Ts.prim_func
def gemm_dyn_shape(
    A: T.Buffer((N, K), "float32"), B: T.Buffer((K, M), "float32"), C: T.Buffer((N, M), "float32")
):
    for i, j, k in T.grid(N, M, K):
        with Ts.sblock("gemm"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dynamic_shape_gemm():
    gemm_dyn_shape_roundtrip = from_source(
        gemm_dyn_shape.script(),
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
    )
    assert_structural_equal_ignore_global_symbol(gemm_dyn_shape, gemm_dyn_shape_roundtrip)


@Ts.prim_func
def buffer_int64(
    A: T.Buffer((T.int64(128), T.int64(128)), dtype="float32"),
    C: T.Buffer((T.int64(128), T.int64(128)), dtype="float32"),
) -> None:
    B = Ts.sblock_alloc_buffer((T.int64(128), T.int64(128)), dtype="float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.prim_func
def buffer_int64_after_roundtrip(
    A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    C: T.Buffer((T.int64(128), T.int64(128)), "float32"),
) -> None:
    B = Ts.sblock_alloc_buffer((T.int64(128), T.int64(128)), dtype="float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


def test_buffer_int64():
    original = buffer_int64
    after_roundtrip = buffer_int64_after_roundtrip
    assert_structural_equal_ignore_global_symbol(original, after_roundtrip, True)


def test_int64_loop():
    @Ts.prim_func
    def int64_grid(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + 1.0

    @Ts.prim_func
    def int64_grid_expanded(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i in range(T.int64(0), T.int64(128)):
            for j in range(T.int64(0), T.int64(128)):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(T.int64(128), i)
                    vj = Ts.axis.spatial(T.int64(128), j)
                    B[vi, vj] = A[vi, vj] + 1.0

    assert_structural_equal_ignore_global_symbol(int64_grid, int64_grid_expanded)


def loop_extent_dependent():
    @Ts.prim_func
    def loop_extent_dependent(A: T.Buffer([], dtype="int32")) -> None:
        for i in T.serial(0, 128):
            for j in T.serial(0, i):
                A[()] = A[()] + j

    return loop_extent_dependent


def parse_bufferslice_as_range_bound():
    # apparently the use of i in the "outer" block when it is defined outside of a block is wrong
    @Ts.prim_func(check_well_formed=False)
    def segment_sum(
        A: T.Buffer([m], dtype="float32"),  # noqa: F821
        B: T.Buffer([n], dtype="float32"),  # noqa: F821
        indptr: T.Buffer([n + 1], dtype="int32"),  # noqa: F821
        n: T.int32,
        m: T.int32,
    ) -> None:
        for i in T.serial(n):
            with Ts.sblock("outer"):
                vi = Ts.axis.spatial(n, i)
                Ts.reads(indptr[i : i + 2], B[vi], A[indptr[i] : indptr[i + 1]])
                Ts.writes(B[vi])
                for j in T.serial(indptr[i], indptr[i + 1]):
                    with Ts.sblock("inner"):
                        vj = Ts.axis.reduce(m, j)
                        Ts.reads(B[vi], A[vj])
                        Ts.writes(B[vi])
                        with Ts.init():
                            B[vi] = T.float32(0)
                        B[vi] = B[vi] + A[vj]

    return segment_sum


def undefined_shape_in_decl_buffer():
    # uninitialized var
    size = T.dynamic("size", "int32")

    @Ts.prim_func(check_well_formed=False)
    def func():
        buf = T.decl_buffer(shape=[size], dtype="float32")
        T.evaluate(buf[0])

    return func


def undefined_stride_in_decl_buffer():
    # uninitialized var
    stride = T.dynamic("stride", "int32")

    @Ts.prim_func(check_well_formed=False)
    def func():
        data_ptr = T.handle("float32")
        buf = T.decl_buffer(shape=[1], dtype="float32", data=data_ptr, strides=[stride])
        T.evaluate(buf[0])

    return func


def undefined_elem_offset_in_decl_buffer():
    # uninitialized var
    elem_offset = T.dynamic("elem_offset", "int32")

    @Ts.prim_func(check_well_formed=False)
    def func():
        data_ptr = T.handle("float32")
        buf = T.decl_buffer(shape=[1], dtype="float32", data=data_ptr, elem_offset=elem_offset)
        T.evaluate(buf[0])

    return func


@pytest.mark.parametrize(
    "ir_generator",
    [
        loop_extent_dependent,
        parse_bufferslice_as_range_bound,
        undefined_shape_in_decl_buffer,
        undefined_stride_in_decl_buffer,
        undefined_elem_offset_in_decl_buffer,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_dynamic_shape(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
