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
# pylint: disable=missing-docstring
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.testing.tir import mma_schedule
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_f16_A_INTRIN,
    LDMATRIX_f16_B_INTRIN,
    LDMATRIX_f16_B_TRANS_INTRIN,
    LDMATRIX_i8_A_INTRIN,
    LDMATRIX_i8_B_TRANS_INTRIN,
    LDMATRIX_i8_B_INTRIN,
    LDMATRIX_e4m3_A_INTRIN,
    LDMATRIX_e4m3_B_TRANS_INTRIN,
    LDMATRIX_e5m2_A_INTRIN,
    LDMATRIX_e5m2_B_TRANS_INTRIN,
    MMA_f16f16f16_INTRIN,
    MMA_f16f16f16_TRANS_B_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_f16f16f32_TRANS_B_INTRIN,
    MMA_fill_16x16_f16_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_fill_16x16_i32_INTRIN,
    MMA_i8i8i32_INTRIN,
    MMA_i8i8i32_TRANS_B_INTRIN,
    MMA_e5m2e5m2f32_TRANS_B_INTRIN,
    MMA_e4m3e4m3f32_TRANS_B_INTRIN,
    MMA_store_16x16_f16_global_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_i32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
    shared_16x32_to_ldmatrix_32x16_layout,
    shared_32x16_to_ldmatrix_32x16_layout,
)

M = 4096
N = 4096
K = 4096
measure_perf = False
gflops = (N * M * K) * 2 / 1e9


def matmul(m, n, k, in_dtype, out_dtype, b_transposed):
    b_shape = (n, k) if b_transposed else (k, n)
    a = te.placeholder((m, k), name="A", dtype=in_dtype)
    b = te.placeholder(b_shape, name="B", dtype=in_dtype)
    k = te.reduce_axis((0, k), name="k")

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return tvm.tir.Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    c = te.compute(
        (m, n),
        lambda i, j: te.sum(maybe_cast(a[i, k]) * maybe_cast(b[maybe_swap(k, j)]), axis=[k]),
        name="C",
    )
    return (a, b, c)


def run_test(
    k_inner,
    in_dtype,
    out_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    index_map_A,
    index_map_B,
    index_map_C,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    mma_intrin,
    mma_fill_intrin,
    mma_store_intrin,
):
    sch = mma_schedule(
        te.create_prim_func(matmul(M, N, K, in_dtype, out_dtype, b_transposed)),
        k_inner,
        in_dtype,
        b_transposed,
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        ldmatrix_a_intrin,
        ldmatrix_b_intrin,
        mma_intrin,
        mma_fill_intrin,
        mma_store_intrin,
    )

    f = tvm.build(sch.mod["main"], target="cuda", name="dense")

    dev = tvm.device("cuda", 0)

    if in_dtype == "float16":
        a_np = np.random.normal(size=(M, K)).astype("float16")

        if b_transposed:
            b_np = np.random.normal(size=(N, K)).astype("float16")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                out_dtype
            )
        else:
            b_np = np.random.normal(size=(K, N)).astype("float16")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype(out_dtype)
    elif in_dtype in ["e4m3_float8", "e5m2_float8"]:
        typemap = {
            "e4m3_float8": "float8_e4m3fn",
            "e5m2_float8": "float8_e5m2",
        }
        a_np = (
            np.random.uniform(low=-5, high=5, size=(M * K))
            .reshape((M, K))
            .astype(typemap[in_dtype])
        )
        if b_transposed:
            b_np = (
                np.random.uniform(low=-5, high=5, size=(N * K))
                .reshape((N, K))
                .astype(typemap[in_dtype])
            )
            c_np = np.dot(a_np.astype("float32"), b_np.T.astype("float32")).astype(out_dtype)
        else:
            b_np = (
                np.random.uniform(low=-5, high=5, size=(N * K))
                .reshape((K, N))
                .astype(typemap[in_dtype])
            )
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype(out_dtype)
    else:
        a_np = np.random.randint(-128, 128, (M, K)).astype("int8")

        if b_transposed:
            b_np = np.random.randint(-128, 128, (N, K)).astype("int8")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                "int32"
            )
        else:
            b_np = np.random.randint(-128, 128, (K, N)).astype("int8")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype("int32")

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype=out_dtype), dev)

    f(a, b, c)

    if out_dtype != "float16" and in_dtype not in ["e4m3_float8", "e5m2_float8"]:
        # The numpy reference is computed with fp32 precision (otherwise too slow).
        # So there is non-trivial accuracy difference if TVM result is computed with fp16 accumulation.
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-2, atol=1e-2)

    return lambda: f.time_evaluator(f.entry_name, dev, number=500)(a, b, c)


@tvm.testing.requires_cuda_compute_version(8)
def test_f16f16f32_m16n16k16():
    def index_map(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    k_inner = 16
    in_dtype = "float16"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [4, 8, 2, 4, 1], [1, 64, 2, 1, 2], [128, 2, 1]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map,
        index_map,
        index_map,
        LDMATRIX_f16_A_INTRIN,
        LDMATRIX_f16_B_INTRIN,
        MMA_f16f16f32_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        True,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map,
        index_map,
        index_map,
        LDMATRIX_f16_A_INTRIN,
        LDMATRIX_f16_B_TRANS_INTRIN,
        MMA_f16f16f32_TRANS_B_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m16n16k16_trans: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_cuda_compute_version(8)
def test_f16f16f16_m16n16k16():
    def index_map(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    k_inner = 16
    in_dtype = "float16"
    out_dtype = "float16"
    i_factors, j_factors, k_factors = [16, 2, 1, 4, 2], [16, 2, 2, 1, 4], [128, 2, 1]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map,
        index_map,
        index_map,
        LDMATRIX_f16_A_INTRIN,
        LDMATRIX_f16_B_INTRIN,
        MMA_f16f16f16_INTRIN,
        MMA_fill_16x16_f16_INTRIN,
        MMA_store_16x16_f16_global_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f16_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        True,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map,
        index_map,
        index_map,
        LDMATRIX_f16_A_INTRIN,
        LDMATRIX_f16_B_TRANS_INTRIN,
        MMA_f16f16f16_TRANS_B_INTRIN,
        MMA_fill_16x16_f16_INTRIN,
        MMA_store_16x16_f16_global_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f16_m16n16k16_trans: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_cuda_compute_version(8)
def test_i8i8i32_m16n16k32():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 32,
            *shared_16x32_to_ldmatrix_32x16_layout(i % 16, j % 32),
        )

    def index_map_B(i, j):
        return (
            i // 32,
            j // 16,
            *shared_32x16_to_ldmatrix_32x16_layout(i % 32, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    k_inner = 32
    in_dtype = "int8"
    out_dtype = "int32"
    i_factors, j_factors, k_factors = [1, 32, 1, 4, 2], [8, 4, 4, 2, 1], [32, 2, 2]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        LDMATRIX_i8_A_INTRIN,
        LDMATRIX_i8_B_INTRIN,
        MMA_i8i8i32_INTRIN,
        MMA_fill_16x16_i32_INTRIN,
        MMA_store_16x16_i32_global_INTRIN,
    )

    if measure_perf and timer:
        print("i8i8i32_m16n16k32: %f GOPS" % (gflops / (timer().mean)))

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        True,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_A,
        index_map_C,
        LDMATRIX_i8_A_INTRIN,
        LDMATRIX_i8_B_TRANS_INTRIN,
        MMA_i8i8i32_TRANS_B_INTRIN,
        MMA_fill_16x16_i32_INTRIN,
        MMA_store_16x16_i32_global_INTRIN,
    )

    if measure_perf and timer:
        print("i8i8i32_m16n16k32_trans: %f GOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_cuda_compute_version(8, 9)
def test_e4m3e4m3f32_m16n16k32():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 32,
            *shared_16x32_to_ldmatrix_32x16_layout(i % 16, j % 32),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    k_inner = 32
    in_dtype = "e4m3_float8"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [1, 32, 1, 4, 2], [8, 4, 4, 2, 1], [32, 2, 2]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        True,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_A,
        index_map_C,
        LDMATRIX_e4m3_A_INTRIN,
        LDMATRIX_e4m3_B_TRANS_INTRIN,
        MMA_e4m3e4m3f32_TRANS_B_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
    )

    if measure_perf and timer:
        print("e4m3e4m3f32_m16n16k32_trans: %f GOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_cuda_compute_version(8, 9)
def test_e5m2e5m2f32_m16n16k32():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 32,
            *shared_16x32_to_ldmatrix_32x16_layout(i % 16, j % 32),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    k_inner = 32
    in_dtype = "e5m2_float8"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [1, 32, 1, 4, 2], [8, 4, 4, 2, 1], [32, 2, 2]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        True,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_A,
        index_map_C,
        LDMATRIX_e5m2_A_INTRIN,
        LDMATRIX_e5m2_B_TRANS_INTRIN,
        MMA_e5m2e5m2f32_TRANS_B_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
    )

    if measure_perf and timer:
        print("e5m2e5m2f32_m16n16k32_trans: %f GOPS" % (gflops / (timer().mean)))


if __name__ == "__main__":
    tvm.testing.main()
