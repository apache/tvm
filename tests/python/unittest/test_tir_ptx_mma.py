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

import sys
import pytest

import tvm
from tvm.script import tir as T
import numpy as np
import tvm.testing


@T.prim_func
def gemm_mma_m8n8k4_row_col_fp64pf64fp64(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 4], dtype="float64")
    B = T.match_buffer(b, [8, 4], dtype="float64")
    C = T.match_buffer(c, [8, 8], dtype="float64")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([1], "float64", scope="local")
    MultiB = T.decl_buffer([1], "float64", scope="local")
    Accum = T.decl_buffer([2], "float64", scope="local")
    for i in range(2):
        Accum[i] = T.float64(0)

    MultiA[0] = A[(tx % 32) // 4, (tx % 32) % 4]
    MultiB[0] = B[(tx % 32) // 4, (tx % 32) % 4]
    T.evaluate(
        T.ptx_mma(
            "m8n8k4",
            "row",
            "col",
            "fp64",
            "fp64",
            "fp64",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="float64",
        )
    )
    for mma_accum_c_id in range(2):
        C[(tx % 32) // 4, (tx % 32) % 4 * 2 + mma_accum_c_id] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m8n8k4_row_col_fp64pf64fp64():
    sch = tvm.tir.Schedule(gemm_mma_m8n8k4_row_col_fp64pf64fp64)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-1, 1, [8, 4]).astype("float64")
    B_np = np.random.uniform(-1, 1, [8, 4]).astype("float64")
    C_np = np.zeros([8, 8]).astype("float64")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float64"), B_np.astype("float64").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m8n8k4_row_row_fp16fp16fp16(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 4], dtype="float16")
    B = T.match_buffer(b, [4, 16], dtype="float16")
    C = T.match_buffer(c, [16, 16], dtype="float16")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([4], "float16", scope="local")
    MultiB = T.decl_buffer([4], "float16", scope="local")
    Accum = T.decl_buffer([8], "float16", scope="local")
    for i in range(8):
        Accum[i] = T.float32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[
            ((tx % 32) % 4) + (4 * ((((tx % 32) // 16 + (tx % 32) % 16 // 4 * 2)) % 4)),
            mma_multi_a_col,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) % 4,
            mma_multi_b_col + (4 * ((tx % 32) // 8)),
        ]
    T.evaluate(
        T.ptx_mma(
            "m8n8k4",
            "row",
            "row",
            "fp16",
            "fp16",
            "fp16",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="float16",
        )
    )
    for mma_accum_c_id in range(8):
        C[
            ((tx % 32) % 4) + (4 * ((((tx % 32) // 16 + (tx % 32) % 16 // 4 * 2)) % 4)),
            mma_accum_c_id % 4 + (4 * ((tx % 32) % 16 // 8)) + mma_accum_c_id // 4 * 8,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(7)
def test_gemm_mma_m8n8k4_row_row_fp16fp16fp16():
    sch = tvm.tir.Schedule(gemm_mma_m8n8k4_row_row_fp16fp16fp16)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-1, 1, [16, 4]).astype("float16")
    B_np = np.random.uniform(-1, 1, [4, 16]).astype("float16")
    C_np = np.zeros([16, 16]).astype("float16")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float16"), B_np.astype("float16"))

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m8n8k4_row_row_fp16fp16fp32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 4], dtype="float16")
    B = T.match_buffer(b, [4, 16], dtype="float16")
    C = T.match_buffer(c, [16, 16], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([4], "float16", scope="local")
    MultiB = T.decl_buffer([4], "float16", scope="local")
    Accum = T.decl_buffer([8], "float32", scope="local")

    for i in range(8):
        Accum[i] = T.float32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[
            ((tx % 32) % 4) + (4 * ((((tx % 32) // 16 + (tx % 32) % 16 // 4 * 2)) % 4)),
            mma_multi_a_col,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) % 4,
            mma_multi_b_col + (4 * ((tx % 32) // 8)),
        ]
    T.evaluate(
        T.ptx_mma(
            "m8n8k4",
            "row",
            "row",
            "fp16",
            "fp16",
            "fp32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="float32",
        )
    )
    for mma_accum_c_id in range(8):
        C[
            ((tx % 32) % 2)
            + ((mma_accum_c_id // 2 % 2) * 2)
            + 4 * ((tx % 32) // 16)
            + ((tx % 32) % 16 // 4) % 2 * 8,
            (tx % 32) % 4 // 2 * 2
            + (tx % 32) % 16 // 8 * 4
            + mma_accum_c_id % 2
            + mma_accum_c_id // 4 * 8,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(7)
def test_gemm_mma_m8n8k4_row_row_fp16fp16fp32():
    sch = tvm.tir.Schedule(gemm_mma_m8n8k4_row_row_fp16fp16fp32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-1, 1, [16, 4]).astype("float16")
    B_np = np.random.uniform(-1, 1, [4, 16]).astype("float16")
    C_np = np.zeros([16, 16]).astype("float32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float32"), B_np.astype("float32"))

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m8n8k16_row_col_s8s8s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 16], dtype="int8")
    B = T.match_buffer(b, [8, 16], dtype="int8")
    C = T.match_buffer(c, [8, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([4], "int8", scope="local")
    MultiB = T.decl_buffer([4], "int8", scope="local")
    Accum = T.decl_buffer([2], "int32", scope="local")
    for i in range(2):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[(tx % 32) // 4, mma_multi_a_col + (tx % 32) % 4 * 4]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[(tx % 32) // 4, mma_multi_b_col + (tx % 32) % 4 * 4]
    T.evaluate(
        T.ptx_mma(
            "m8n8k16",
            "row",
            "col",
            "int8",
            "int8",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(2):
        C[(tx % 32) // 4, (tx % 32) % 4 * 2 + mma_accum_c_id] = Accum[mma_accum_c_id]


# This test uses mma instructions that are not available on NVCC 10.1.
# Failure occurs during the external call to nvcc, when attempting to
# generate the .fatbin file.
@tvm.testing.requires_nvcc_version(11)
@tvm.testing.requires_cuda_compute_version(7, 5)
def test_gemm_mma_m8n8k16_row_col_s8s8s32():
    sch = tvm.tir.Schedule(gemm_mma_m8n8k16_row_col_s8s8s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-10, 10, [8, 16]).astype("int8")
    B_np = np.random.uniform(-10, 10, [8, 16]).astype("int8")
    C_np = np.zeros([8, 8]).astype("int32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int32"), B_np.astype("int32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m8n8k16_row_col_s8u8s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 16], dtype="int8")
    B = T.match_buffer(b, [8, 16], dtype="uint8")
    C = T.match_buffer(c, [8, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([4], "int8", scope="local")
    MultiB = T.decl_buffer([4], "uint8", scope="local")
    Accum = T.decl_buffer([2], "int32", scope="local")
    for i in range(2):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[(tx % 32) // 4, mma_multi_a_col + (tx % 32) % 4 * 4]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[(tx % 32) // 4, mma_multi_b_col + (tx % 32) % 4 * 4]
    T.evaluate(
        T.ptx_mma(
            "m8n8k16",
            "row",
            "col",
            "int8",
            "uint8",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(2):
        C[(tx % 32) // 4, (tx % 32) % 4 * 2 + mma_accum_c_id] = Accum[mma_accum_c_id]


# This test uses mma instructions that are not available on NVCC 10.1.
# Failure occurs during the external call to nvcc, when attempting to
# generate the .fatbin file.
@tvm.testing.requires_nvcc_version(11)
@tvm.testing.requires_cuda_compute_version(7, 5)
def test_gemm_mma_m8n8k16_row_col_s8u8s32():
    sch = tvm.tir.Schedule(gemm_mma_m8n8k16_row_col_s8u8s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-10, 10, [8, 16]).astype("int8")
    B_np = np.random.uniform(-10, 10, [8, 16]).astype("uint8")
    C_np = np.zeros([8, 8]).astype("int32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int32"), B_np.astype("int32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m8n8k32_row_col_s4s4s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 32], dtype="int4")
    B = T.match_buffer(b, [8, 32], dtype="int4")
    C = T.match_buffer(c, [8, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([8], "int4", scope="local")
    MultiB = T.decl_buffer([8], "int4", scope="local")
    Accum = T.decl_buffer([2], "int32", scope="local")
    for i in range(2):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in T.vectorized(8):
        MultiA[mma_multi_a_col] = A[(tx % 32) // 4, mma_multi_a_col + (tx % 32) % 4 * 8]
    for mma_multi_b_col in T.vectorized(8):
        MultiB[mma_multi_b_col] = B[(tx % 32) // 4, mma_multi_b_col + (tx % 32) % 4 * 8]
    T.evaluate(
        T.ptx_mma(
            "m8n8k32",
            "row",
            "col",
            "int4",
            "int4",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(2):
        C[(tx % 32) // 4, (tx % 32) % 4 * 2 + mma_accum_c_id] = Accum[mma_accum_c_id]


# This test uses mma instructions that are not available on NVCC 10.1.
# Failure occurs during the external call to nvcc, when attempting to
# generate the .fatbin file.
@tvm.testing.requires_nvcc_version(11)
@tvm.testing.requires_cuda_compute_version(7, 5)
def test_gemm_mma_m8n8k32_row_col_s4s4s32():
    sch = tvm.tir.Schedule(gemm_mma_m8n8k32_row_col_s4s4s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.empty([8, 32], "int4", ctx)
    B_tvm = tvm.nd.empty([8, 32], "int4", ctx)
    C_tvm = tvm.nd.empty([8, 8], "int32", ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)
    # Currently the correctness is not checked.
    # TODO: add correctness checking here.


@T.prim_func
def gemm_mma_m8n8k32_row_col_s4u4s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [8, 32], dtype="int4")
    B = T.match_buffer(b, [8, 32], dtype="uint4")
    C = T.match_buffer(c, [8, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([8], "int4", scope="local")
    MultiB = T.decl_buffer([8], "uint4", scope="local")
    Accum = T.decl_buffer([2], "int32", scope="local")
    for i in range(2):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in T.vectorized(8):
        MultiA[mma_multi_a_col] = A[(tx % 32) // 4, mma_multi_a_col + (tx % 32) % 4 * 8]
    for mma_multi_b_col in T.vectorized(8):
        MultiB[mma_multi_b_col] = B[(tx % 32) // 4, mma_multi_b_col + (tx % 32) % 4 * 8]
    T.evaluate(
        T.ptx_mma(
            "m8n8k32",
            "row",
            "col",
            "int4",
            "uint4",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(2):
        C[(tx % 32) // 4, (tx % 32) % 4 * 2 + mma_accum_c_id] = Accum[mma_accum_c_id]


# This test uses mma instructions that are not available on NVCC 10.1.
# Failure occurs during the external call to nvcc, when attempting to
# generate the .fatbin file.
@tvm.testing.requires_nvcc_version(11)
@tvm.testing.requires_cuda_compute_version(7, 5)
def test_gemm_mma_m8n8k32_row_col_s4u4s32():
    sch = tvm.tir.Schedule(gemm_mma_m8n8k32_row_col_s4u4s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.empty([8, 32], "int4", ctx)
    B_tvm = tvm.nd.empty([8, 32], "uint4", ctx)
    C_tvm = tvm.nd.empty([8, 8], "int32", ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)
    # Currently the correctness is not checked.
    # TODO: add correctness checking here.


@T.prim_func
def gemm_mma_m16n8k8_row_col_fp16fp16fp32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 8], dtype="float16")
    B = T.match_buffer(b, [8, 8], dtype="float16")
    C = T.match_buffer(c, [16, 8], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([4], "float16", scope="local")
    MultiB = T.decl_buffer([2], "float16", scope="local")
    Accum = T.decl_buffer([4], "float32", scope="local")
    for i in range(4):
        Accum[i] = T.float32(0)

    for mma_multi_a_col in T.vectorized(4):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col // 2 * 8, (tx % 32) % 4 * 2 + mma_multi_a_col % 2
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4 + mma_multi_b_col // 2 * 8, (tx % 32) % 4 * 2 + mma_multi_b_col % 2
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k8",
            "row",
            "col",
            "fp16",
            "fp16",
            "fp32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="float32",
        )
    )
    for mma_accum_c_id in range(4):
        C[(tx % 32) // 4 + mma_accum_c_id // 2 * 8, (tx % 32) % 4 * 2 + mma_accum_c_id % 2] = Accum[
            mma_accum_c_id
        ]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k8_row_col_fp16fp16fp32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k8_row_col_fp16fp16fp32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-1, 1, [16, 8]).astype("float16")
    B_np = np.random.uniform(-1, 1, [8, 8]).astype("float16")
    C_np = np.zeros([16, 8]).astype("float32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float32"), B_np.astype("float32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m16n8k16_row_col_fp16fp16fp16(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="float16")
    B = T.match_buffer(b, [8, 16], dtype="float16")
    C = T.match_buffer(c, [16, 8], dtype="float16")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([8], "float16", scope="local")
    MultiB = T.decl_buffer([4], "float16", scope="local")
    Accum = T.decl_buffer([4], "float16", scope="local")
    for i in range(4):
        Accum[i] = T.float32(0)

    for mma_multi_a_col in range(8):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col % 4 // 2 * 8,
            (tx % 32) % 4 * 2 + mma_multi_a_col % 2 + mma_multi_a_col // 4 * 8,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 2 + mma_multi_b_col % 2 + mma_multi_b_col // 2 * 8,
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k16",
            "row",
            "col",
            "fp16",
            "fp16",
            "fp16",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="float16",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k16_row_col_fp16fp16fp16():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k16_row_col_fp16fp16fp16)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-1, 1, [16, 16]).astype("float16")
    B_np = np.random.uniform(-1, 1, [8, 16]).astype("float16")
    C_np = np.zeros([16, 8]).astype("float16")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float16"), B_np.astype("float16").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m16n8k16_row_col_fp16fp16fp32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="float16")
    B = T.match_buffer(b, [8, 16], dtype="float16")
    C = T.match_buffer(c, [16, 8], dtype="float32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([8], "float16", scope="local")
    MultiB = T.decl_buffer([4], "float16", scope="local")
    Accum = T.decl_buffer([4], "float32", scope="local")
    for i in range(4):
        Accum[i] = T.float32(0)

    for mma_multi_a_col in range(8):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col % 4 // 2 * 8,
            (tx % 32) % 4 * 2 + mma_multi_a_col % 2 + mma_multi_a_col // 4 * 8,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 2 + mma_multi_b_col % 2 + mma_multi_b_col // 2 * 8,
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k16",
            "row",
            "col",
            "fp16",
            "fp16",
            "fp32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="float32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k16_row_col_fp16fp16fp32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k16_row_col_fp16fp16fp32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-1, 1, [16, 16]).astype("float16")
    B_np = np.random.uniform(-1, 1, [8, 16]).astype("float16")
    C_np = np.zeros([16, 8]).astype("float32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("float32"), B_np.astype("float32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m16n8k16_row_col_s8s8s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="int8")
    B = T.match_buffer(b, [8, 16], dtype="int8")
    C = T.match_buffer(c, [16, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([8], "int8", scope="local")
    MultiB = T.decl_buffer([4], "int8", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")
    for i in range(4):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in range(8):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col // 4 * 8,
            (tx % 32) % 4 * 4 + mma_multi_a_col % 4,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 4 + mma_multi_b_col,
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k16",
            "row",
            "col",
            "int8",
            "int8",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k16_row_col_s8s8s32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k16_row_col_s8s8s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-10, 10, [16, 16]).astype("int8")
    B_np = np.random.uniform(-10, 10, [8, 16]).astype("int8")
    C_np = np.zeros([16, 8]).astype("int32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int32"), B_np.astype("int32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m16n8k16_row_col_s8u8s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="int8")
    B = T.match_buffer(b, [8, 16], dtype="uint8")
    C = T.match_buffer(c, [16, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([8], "int8", scope="local")
    MultiB = T.decl_buffer([4], "uint8", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")
    for i in range(4):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in range(8):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col // 4 * 8,
            (tx % 32) % 4 * 4 + mma_multi_a_col % 4,
        ]
    for mma_multi_b_col in T.vectorized(4):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 4 + mma_multi_b_col,
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k16",
            "row",
            "col",
            "int8",
            "uint8",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k16_row_col_s8u8s32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k16_row_col_s8u8s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-10, 10, [16, 16]).astype("int8")
    B_np = np.random.uniform(-10, 10, [8, 16]).astype("uint8")
    C_np = np.zeros([16, 8]).astype("int32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int32"), B_np.astype("int32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m16n8k32_row_col_s8s8s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 32], dtype="int8")
    B = T.match_buffer(b, [8, 32], dtype="int8")
    C = T.match_buffer(c, [16, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([16], "int8", scope="local")
    MultiB = T.decl_buffer([8], "int8", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")
    for i in range(4):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in range(16):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col % 8 // 4 * 8,
            (tx % 32) % 4 * 4 + mma_multi_a_col % 4 + mma_multi_a_col // 8 * 16,
        ]
    for mma_multi_b_col in range(8):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 4 + mma_multi_b_col % 4 + mma_multi_b_col // 4 * 16,
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k32",
            "row",
            "col",
            "int8",
            "int8",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k32_row_col_s8s8s32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k32_row_col_s8s8s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-10, 10, [16, 32]).astype("int8")
    B_np = np.random.uniform(-10, 10, [8, 32]).astype("int8")
    C_np = np.zeros([16, 8]).astype("int32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int32"), B_np.astype("int32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m16n8k32_row_col_s8u8s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 32], dtype="int8")
    B = T.match_buffer(b, [8, 32], dtype="uint8")
    C = T.match_buffer(c, [16, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([16], "int8", scope="local")
    MultiB = T.decl_buffer([8], "uint8", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")
    for i in range(4):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in range(16):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col % 8 // 4 * 8,
            (tx % 32) % 4 * 4 + mma_multi_a_col % 4 + mma_multi_a_col // 8 * 16,
        ]
    for mma_multi_b_col in range(8):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 4 + mma_multi_b_col % 4 + mma_multi_b_col // 4 * 16,
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k32",
            "row",
            "col",
            "int8",
            "uint8",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k32_row_col_s8u8s32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k32_row_col_s8u8s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(-10, 10, [16, 32]).astype("int8")
    B_np = np.random.uniform(-10, 10, [8, 32]).astype("uint8")
    C_np = np.zeros([16, 8]).astype("int32")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)

    golden = np.matmul(A_np.astype("int32"), B_np.astype("int32").T)

    C_numpy = C_tvm.numpy()

    tvm.testing.assert_allclose(golden, C_numpy, atol=1e-3, rtol=1e-3)


@T.prim_func
def gemm_mma_m16n8k64_row_col_s4s4s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 64], dtype="int4")
    B = T.match_buffer(b, [8, 64], dtype="int4")
    C = T.match_buffer(c, [16, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([32], "int4", scope="local")
    MultiB = T.decl_buffer([16], "int4", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")
    for i in range(4):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in range(32):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col % 16 // 8 * 8,
            (tx % 32) % 4 * 8 + mma_multi_a_col % 8 + mma_multi_a_col // 16 * 32,
        ]
    for mma_multi_b_col in range(16):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 8 + mma_multi_b_col % 8 + mma_multi_b_col // 8 * 32,
        ]
    T.evaluate(
        T.ptx_mma(
            "m8n8k32",
            "row",
            "col",
            "int4",
            "int4",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k64_row_col_s4s4s32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k64_row_col_s4s4s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.empty([16, 64], "int4", ctx)
    B_tvm = tvm.nd.empty([8, 64], "int4", ctx)
    C_tvm = tvm.nd.empty([16, 8], "int32", ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)
    # Currently the correctness is not checked.
    # TODO: add correctness checking here.


@T.prim_func
def gemm_mma_m16n8k64_row_col_s4u4s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 64], dtype="int4")
    B = T.match_buffer(b, [8, 64], dtype="uint4")
    C = T.match_buffer(c, [16, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([32], "int4", scope="local")
    MultiB = T.decl_buffer([16], "uint4", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")
    for i in range(4):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in range(32):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col % 16 // 8 * 8,
            (tx % 32) % 4 * 8 + mma_multi_a_col % 8 + mma_multi_a_col // 16 * 32,
        ]
    for mma_multi_b_col in range(16):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 8 + mma_multi_b_col % 8 + mma_multi_b_col // 8 * 32,
        ]
    T.evaluate(
        T.ptx_mma(
            "m8n8k32",
            "row",
            "col",
            "int4",
            "uint4",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k64_row_col_s4u4s32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k64_row_col_s4u4s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.empty([16, 64], "int4", ctx)
    B_tvm = tvm.nd.empty([8, 64], "uint4", ctx)
    C_tvm = tvm.nd.empty([16, 8], "int32", ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)
    # Currently the correctness is not checked.
    # TODO: add correctness checking here.


@T.prim_func
def gemm_mma_m16n8k256_row_col_b1b1s32(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 256], dtype="int1")
    B = T.match_buffer(b, [8, 256], dtype="int1")
    C = T.match_buffer(c, [16, 8], dtype="int32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    MultiA = T.decl_buffer([128], "int1", scope="local")
    MultiB = T.decl_buffer([64], "int1", scope="local")
    Accum = T.decl_buffer([4], "int32", scope="local")
    for i in range(4):
        Accum[i] = T.int32(0)

    for mma_multi_a_col in range(128):
        MultiA[mma_multi_a_col] = A[
            (tx % 32) // 4 + mma_multi_a_col % 64 // 32 * 8,
            (tx % 32) % 4 * 32 + mma_multi_a_col % 32 + mma_multi_a_col // 64 * 128,
        ]
    for mma_multi_b_col in range(16):
        MultiB[mma_multi_b_col] = B[
            (tx % 32) // 4,
            (tx % 32) % 4 * 32 + mma_multi_b_col % 32 + mma_multi_b_col // 32 * 128,
        ]
    T.evaluate(
        T.ptx_mma(
            "m16n8k256",
            "row",
            "col",
            "int1",
            "int1",
            "int32",
            MultiA.data,
            0,
            MultiB.data,
            0,
            Accum.data,
            0,
            False,
            "xor",
            dtype="int32",
        )
    )
    for mma_accum_c_id in range(4):
        C[
            (tx % 32) // 4 + mma_accum_c_id // 2 * 8,
            (tx % 32) % 4 * 2 + mma_accum_c_id % 2,
        ] = Accum[mma_accum_c_id]


@tvm.testing.requires_cuda_compute_version(8)
def test_gemm_mma_m16n8k256_row_col_b1b1s32():
    sch = tvm.tir.Schedule(gemm_mma_m16n8k256_row_col_b1b1s32)
    cuda_mod = tvm.build(sch.mod, target="cuda")

    ctx = tvm.cuda()
    A_tvm = tvm.nd.empty([16, 256], "int1", ctx)
    B_tvm = tvm.nd.empty([8, 256], "int1", ctx)
    C_tvm = tvm.nd.empty([16, 8], "int32", ctx)

    cuda_mod(A_tvm, B_tvm, C_tvm)
    # Currently the correctness is not checked.
    # TODO: add correctness checking here.


if __name__ == "__main__":
    tvm.testing.main()
