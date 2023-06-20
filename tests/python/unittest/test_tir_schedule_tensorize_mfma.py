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
import tvm
from tvm import te
from tvm.tir.tensor_intrin.rocm import (
    shared_16x4_to_local_64x1_layout_A,
    shared_4x16_to_local_64x1_layout_B,
    shared_16x16_to_local_64x4_layout_A,
    shared_16x16_to_local_64x4_layout_B,
    shared_16x16_to_local_64x4_layout_C,
    ROCM_MFMA_fill_16x16_f32_INTRIN,
    ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN,
    ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
    ROCM_MFMA_f32f32f32_INTRIN,
    ROCM_MFMA_STORE_16x16_f32_INTRIN,
    ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN,
    ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
    ROCM_MFMA_f16f16f32_INTRIN,
    ROCM_MFMA_STORE_16x16_f32_INTRIN,
    ROCM_MFMA_fill_16x16_i32_INTRIN,
    ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN,
    ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN,
    ROCM_MFMA_s8s8s32_INTRIN,
    ROCM_MFMA_STORE_16x16_s32_INTRIN,
)
import tvm.testing
import numpy as np
from tvm.testing.tir import mfma_schedule


M = 1024
N = 1024
K = 1024
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
    sch = mfma_schedule(
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

    f = tvm.build(sch.mod["main"], target="rocm", name="dense")

    dev = tvm.device("rocm", 0)
    if in_dtype == "float32":
        a_np = np.random.uniform(size=(M, K)).astype("float32")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float32")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                out_dtype
            )
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float32")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype(out_dtype)
    elif in_dtype == "float16":
        a_np = np.random.uniform(size=(M, K)).astype("float16")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float16")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                out_dtype
            )
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float16")
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

    if in_dtype != "float16":
        # The numpy reference is computed with fp32 precision (otherwise too slow).
        # So there is non-trivial accuracy difference if TVM result is computed with fp16 accumulation.
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-2, atol=1e-2)

    return lambda: f.time_evaluator(f.entry_name, dev, number=500)(a, b, c)


@tvm.testing.requires_matrixcore
def test_i8i8i32_m16n16k16():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_A(i % 16, j % 16),
        )

    def index_map_B(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_B(i % 16, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16),
        )

    k_inner = 16
    in_dtype = "int8"
    out_dtype = "int32"
    i_factors, j_factors, k_factors = [1, 8, 2, 4, 1], [1, 16, 2, 1, 2], [32, 2, 1]

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
        ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN,
        ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN,
        ROCM_MFMA_s8s8s32_INTRIN,
        ROCM_MFMA_fill_16x16_i32_INTRIN,
        ROCM_MFMA_STORE_16x16_s32_INTRIN,
    )

    if measure_perf and timer:
        print("test_i8i8i32_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_matrixcore
def test_f16f16f32_m16n16k16():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_A(i % 16, j % 16),
        )

    def index_map_B(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_B(i % 16, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16),
        )

    k_inner = 16
    in_dtype = "float16"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [1, 8, 2, 4, 1], [1, 16, 2, 1, 2], [32, 2, 1]

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
        ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN,
        ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
        ROCM_MFMA_f16f16f32_INTRIN,
        ROCM_MFMA_fill_16x16_f32_INTRIN,
        ROCM_MFMA_STORE_16x16_f32_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_matrixcore
def test_f32f32f32_m16n16k4():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x4_to_local_64x1_layout_A(i % 16, j % 16),
        )

    def index_map_B(i, j):
        return (
            i // 16,
            j // 16,
            *shared_4x16_to_local_64x1_layout_B(i % 16, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16),
        )

    k_inner = 4
    in_dtype = "float32"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [4, 2, 1, 4, 2], [4, 2, 2, 1, 4], [128, 2, 1]

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
        ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN,
        ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
        ROCM_MFMA_f32f32f32_INTRIN,
        ROCM_MFMA_fill_16x16_f32_INTRIN,
        ROCM_MFMA_STORE_16x16_f32_INTRIN,
    )

    if measure_perf and timer:
        print("test_f32f32f32_m16n16k4: %f GFLOPS" % (gflops / (timer().mean)))


if __name__ == "__main__":
    tvm.testing.main()
