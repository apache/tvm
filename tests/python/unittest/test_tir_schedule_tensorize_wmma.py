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
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
    WMMA_FILL_32x32x8_F32_INTRIN,
    WMMA_LOAD_32x32x8_F16_A_INTRIN,
    WMMA_LOAD_32x32x8_F16_B_INTRIN,
    WMMA_SYNC_32x32x8_f16f16f32_INTRIN,
    WMMA_STORE_32x32x8_F32_GLOBAL_INTRIN,
    WMMA_FILL_16x16x16_I32_INTRIN,
    WMMA_LOAD_16x16x16_I8_A_INTRIN,
    WMMA_LOAD_16x16x16_I8_B_INTRIN,
    WMMA_SYNC_16x16x16_I8I8I32_INTRIN,
    WMMA_STORE_16x16x16_I32_GLOBAL_INTRIN,
    WMMA_FILL_32x32x8_I32_INTRIN,
    WMMA_LOAD_32x32x8_I8_A_INTRIN,
    WMMA_LOAD_32x32x8_I8_B_INTRIN,
    WMMA_SYNC_32x32x8_I8I8I32_INTRIN,
    WMMA_STORE_32x32x8_I32_GLOBAL_INTRIN,
)
import tvm.testing
import numpy as np
from tvm.testing.tir import wmma_schedule

M = 1024
N = 1024
K = 1024
measure_perf = True
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


def run_wmma_test(
    k_inner,
    in_dtype,
    out_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    wmma_m,
    wmma_n,
    wmma_k,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    wmma_intrin,
    wmma_fill_intrin,
    wmma_store_intrin,
):

    sch = wmma_schedule(
        te.create_prim_func(matmul(M, N, K, in_dtype, out_dtype, b_transposed)),
        k_inner,
        in_dtype,
        b_transposed,
        i_factors,
        j_factors,
        k_factors,
        wmma_m,
        wmma_n,
        wmma_k,
        ldmatrix_a_intrin,
        ldmatrix_b_intrin,
        wmma_intrin,
        wmma_fill_intrin,
        wmma_store_intrin,
    )

    f = tvm.build(sch.mod["main"], target="hip", name="dense")

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

    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-2, atol=1e-2)

    return lambda: f.time_evaluator(f.entry_name, dev, number=500)(a, b, c)


@tvm.testing.requires_matrixcore
def test_wmma_f16f16f32_m16n16k16():
    k_inner = 16
    in_dtype = "float16"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [1, 8, 2, 4, 1], [1, 16, 2, 1, 2], [32, 2, 1]
    timer = run_wmma_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        16,
        16,
        16,
        WMMA_LOAD_16x16x16_F16_A_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_INTRIN,
        WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
        WMMA_FILL_16x16x16_F32_INTRIN,
        WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_matrixcore
def test_wmma_f16f16f32_m32n32k8():
    k_inner = 8
    in_dtype = "float16"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [1, 4, 2, 4, 1], [1, 8, 2, 1, 2], [64, 2, 1]

    timer = run_wmma_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        32,
        32,
        8,
        WMMA_LOAD_32x32x8_F16_A_INTRIN,
        WMMA_LOAD_32x32x8_F16_B_INTRIN,
        WMMA_SYNC_32x32x8_f16f16f32_INTRIN,
        WMMA_FILL_32x32x8_F32_INTRIN,
        WMMA_STORE_32x32x8_F32_GLOBAL_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m32n32k8: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_matrixcore
def test_wmma_i8i8i32_m16n16k16():
    k_inner = 16
    in_dtype = "int8"
    out_dtype = "int32"
    i_factors, j_factors, k_factors = [1, 8, 2, 4, 1], [1, 16, 2, 1, 2], [32, 2, 1]

    timer = run_wmma_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        16,
        16,
        16,
        WMMA_LOAD_16x16x16_I8_A_INTRIN,
        WMMA_LOAD_16x16x16_I8_B_INTRIN,
        WMMA_SYNC_16x16x16_I8I8I32_INTRIN,
        WMMA_FILL_16x16x16_I32_INTRIN,
        WMMA_STORE_16x16x16_I32_GLOBAL_INTRIN,
    )

    if measure_perf and timer:
        print("i8i8i8_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_matrixcore
def test_wmma_i8i8i32_m32n32k8():
    k_inner = 8
    in_dtype = "int8"
    out_dtype = "int32"
    i_factors, j_factors, k_factors = [1, 4, 2, 4, 1], [1, 8, 2, 1, 2], [64, 2, 1]

    timer = run_wmma_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        32,
        32,
        8,
        WMMA_LOAD_32x32x8_I8_A_INTRIN,
        WMMA_LOAD_32x32x8_I8_B_INTRIN,
        WMMA_SYNC_32x32x8_I8I8I32_INTRIN,
        WMMA_FILL_32x32x8_I32_INTRIN,
        WMMA_STORE_32x32x8_I32_GLOBAL_INTRIN,
    )

    if measure_perf and timer:
        print("i8i8i32_m32n32k8: %f GFLOPS" % (gflops / (timer().mean)))


if __name__ == "__main__":
    test_wmma_f16f16f32_m16n16k16()
