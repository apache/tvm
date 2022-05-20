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
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_INTRIN,
    LDMATRIX_16x16_B_INTRIN,
    LDMATRIX_16x16_B_TRANS_INTRIN,
    LDMATRIX_16x32_A_INTRIN,
    LDMATRIX_32x16_B_INTRIN,
    LDMATRIX_16x32_B_TRANS_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_f16f16f32_TRANS_INTRIN,
    MMA_f16f16f16_INTRIN,
    MMA_f16f16f16_TRANS_INTRIN,
    MMA_i8i8i32_INTRIN,
    MMA_i8i8i32_TRANS_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_fill_16x16_f16_INTRIN,
    MMA_fill_16x16_i32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_f16_global_INTRIN,
    MMA_store_16x16_i32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
    shared_32x16_to_ldmatrix_32x16_layout,
    shared_16x32_to_ldmatrix_32x16_layout,
)
import tvm.testing
import numpy as np


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


def is_ampere_or_newer():
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    return major >= 8


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
    workload = te.create_prim_func(matmul(M, N, K, in_dtype, out_dtype, b_transposed))
    ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(ir_module)

    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 16])
    k, k_tc = sch.split(k, factors=[None, k_inner])

    sch.reorder(i, j, k, i_tc, j_tc, k_tc)

    block_inner = sch.blockize(i_tc)
    block_outer, block_inner = block_inner, block

    num_ty = i_factors[2] * j_factors[2]

    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)

    sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3, k2, i4, j4)

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(j2, i2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, "shared")
        sch.compute_at(block_read, k0)
        vector_size = 16 if in_dtype == "int8" else 8
        warp_size = 32
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
        sch.bind(f_2, "threadIdx.x")
        sch.bind(f_1, "threadIdx.y")
        sch.vectorize(f_3)
        offset = 8 if in_dtype == "float16" else 16
        sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)

        return block_read

    fetch_to_shared(block_outer, 0, 2)
    fetch_to_shared(block_outer, 1, 2)

    A_warp = sch.cache_read(block_outer, 0, "warp")
    B_warp = sch.cache_read(block_outer, 1, "warp")

    sch.compute_at(A_warp, k1)
    sch.compute_at(B_warp, k1)

    C_warp = sch.cache_write(block_outer, 0, "warp")
    sch.reverse_compute_at(C_warp, thread_idy)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])
    block_init_c = sch.get_block("C_init")

    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1

    loop_a = tile_wmma_fragment(A_warp, 16, k_inner)

    if b_transposed:
        loop_b = tile_wmma_fragment(B_warp, 16, k_inner)
    else:
        loop_b = tile_wmma_fragment(B_warp, k_inner, 16)

    sch.transform_layout(A_warp, 0, "write", index_map_A)
    sch.transform_layout(B_warp, 0, "write", index_map_B)
    sch.transform_layout(C_warp, 0, "read", index_map_C)

    sch.tensorize(loop_a, ldmatrix_a_intrin)
    sch.tensorize(loop_b, ldmatrix_b_intrin)
    sch.tensorize(sch.get_loops(block_inner)[-3], mma_intrin)
    sch.tensorize(sch.get_loops(block_init_c)[-2], mma_fill_intrin)
    sch.tensorize(sch.get_loops(C_warp)[-2], mma_store_intrin)

    if not is_ampere_or_newer():
        return None

    f = tvm.build(sch.mod["main"], target="cuda", name="dense")

    dev = tvm.device("cuda", 0)

    if in_dtype == "float16":
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

    if out_dtype != "float16":
        # The numpy reference is computed with fp32 precision (otherwise too slow).
        # So there is non-trivial accuracy difference if TVM result is computed with fp16 accumulation.
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    return lambda: f.time_evaluator(f.entry_name, dev, number=500)(a, b, c)


@tvm.testing.requires_cuda
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
        LDMATRIX_16x16_A_INTRIN,
        LDMATRIX_16x16_B_INTRIN,
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
        LDMATRIX_16x16_A_INTRIN,
        LDMATRIX_16x16_B_TRANS_INTRIN,
        MMA_f16f16f32_TRANS_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m16n16k16_trans: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_cuda
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
        LDMATRIX_16x16_A_INTRIN,
        LDMATRIX_16x16_B_INTRIN,
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
        LDMATRIX_16x16_A_INTRIN,
        LDMATRIX_16x16_B_TRANS_INTRIN,
        MMA_f16f16f16_TRANS_INTRIN,
        MMA_fill_16x16_f16_INTRIN,
        MMA_store_16x16_f16_global_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f16_m16n16k16_trans: %f GFLOPS" % (gflops / (timer().mean)))


@tvm.testing.requires_cuda
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
        LDMATRIX_16x32_A_INTRIN,
        LDMATRIX_32x16_B_INTRIN,
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
        LDMATRIX_16x32_A_INTRIN,
        LDMATRIX_16x32_B_TRANS_INTRIN,
        MMA_i8i8i32_TRANS_INTRIN,
        MMA_fill_16x16_i32_INTRIN,
        MMA_store_16x16_i32_global_INTRIN,
    )

    if measure_perf and timer:
        print("i8i8i32_m16n16k32_trans: %f GOPS" % (gflops / (timer().mean)))


if __name__ == "__main__":
    test_f16f16f32_m16n16k16()
    test_f16f16f16_m16n16k16()
    test_i8i8i32_m16n16k32()
