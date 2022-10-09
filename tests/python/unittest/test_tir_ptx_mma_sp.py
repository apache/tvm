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

import tvm
from tvm.script import tir as T
import numpy as np
import tvm.testing


def gen_2in4_mask(m: int, n: int):
    assert n % 4 == 0
    return np.array(
        [[np.sort(np.random.choice(4, 2, replace=False)) for _ in range(n // 4)] for _ in range(m)]
    ).astype("uint8")


def get_dense_mat_by_mask(val, mask):
    m, n_chunks, _ = mask.shape
    val = val.reshape(m, n_chunks, 2)
    ret = np.zeros((m, n_chunks, 4)).astype(val.dtype)
    for i in range(m):
        for j in range(n_chunks):
            for k in range(2):
                ret[i, j, mask[i, j, k]] = val[i, j, k]
    return ret.reshape(m, n_chunks * 4)


@T.prim_func
def mma_sp_m16n8k16_f16f16f16(a: T.handle, b: T.handle, c: T.handle, _metadata: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 8], dtype="float16")
    B = T.match_buffer(b, [16, 8], dtype="float16")
    C = T.match_buffer(c, [16, 8], dtype="float16")
    metadata = T.match_buffer(_metadata, [8], dtype="uint32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    multi_a = T.decl_buffer([4], "float16", scope="local")
    multi_b = T.decl_buffer([4], "float16", scope="local")
    accum = T.decl_buffer([4], "float16", scope="local")
    meta_local = T.decl_buffer([1], "uint32", scope="local")
    for i in range(4):
        accum[i] = T.float16(0)

    for i in range(4):
        multi_a[i] = A[tx // 4 + i // 2 * 8, tx % 4 * 2 + i % 2]

    for i in range(4):
        multi_b[i] = B[tx % 4 * 2 + i % 2 + i // 2 * 8, tx // 4]

    meta_local[0] = metadata[tx // 4]

    T.evaluate(
        T.ptx_mma_sp(
            "m16n8k16",
            "row",
            "col",
            "fp16",
            "fp16",
            "fp16",
            multi_a.data,
            0,
            multi_b.data,
            0,
            accum.data,
            0,
            meta_local.data,
            0,
            0,
            False,
            dtype="float16",
        )
    )

    for i in range(4):
        C[i // 2 * 8 + tx // 4, tx % 4 * 2 + i % 2] = accum[i]


@T.prim_func
def mma_sp_m16n8k16_f16f16f32(a: T.handle, b: T.handle, c: T.handle, _metadata: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 8], dtype="float16")
    B = T.match_buffer(b, [16, 8], dtype="float16")
    C = T.match_buffer(c, [16, 8], dtype="float32")
    metadata = T.match_buffer(_metadata, [8], dtype="uint32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    multi_a = T.decl_buffer([4], "float16", scope="local")
    multi_b = T.decl_buffer([4], "float16", scope="local")
    accum = T.decl_buffer([4], "float32", scope="local")
    meta_local = T.decl_buffer([1], "uint32", scope="local")
    for i in range(4):
        accum[i] = T.float16(0)

    for i in range(4):
        multi_a[i] = A[tx // 4 + i // 2 * 8, tx % 4 * 2 + i % 2]

    for i in range(4):
        multi_b[i] = B[tx % 4 * 2 + i % 2 + i // 2 * 8, tx // 4]

    meta_local[0] = metadata[tx // 4]

    T.evaluate(
        T.ptx_mma_sp(
            "m16n8k16",
            "row",
            "col",
            "fp16",
            "fp16",
            "fp32",
            multi_a.data,
            0,
            multi_b.data,
            0,
            accum.data,
            0,
            meta_local.data,
            0,
            0,
            False,
            dtype="float32",
        )
    )

    for i in range(4):
        C[i // 2 * 8 + tx // 4, tx % 4 * 2 + i % 2] = accum[i]


@T.prim_func
def mma_sp_m16n8k32_f16f16f16(a: T.handle, b: T.handle, c: T.handle, _metadata: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="float16")
    B = T.match_buffer(b, [32, 8], dtype="float16")
    C = T.match_buffer(c, [16, 8], dtype="float16")
    metadata = T.match_buffer(_metadata, [16], dtype="uint32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    multi_a = T.decl_buffer([8], "float16", scope="local")
    multi_b = T.decl_buffer([8], "float16", scope="local")
    accum = T.decl_buffer([4], "float16", scope="local")
    meta_local = T.decl_buffer([1], "uint32", scope="local")
    for i in range(4):
        accum[i] = T.float16(0)

    for i in range(8):
        multi_a[i] = A[(i % 4) // 2 * 8 + tx // 4, i // 4 * 8 + tx % 4 * 2 + i % 2]

    for i in range(8):
        multi_b[i] = B[i // 2 * 8 + tx % 4 * 2 + i % 2, tx // 4]

    meta_local[0] = metadata[tx // 4 * 2 + tx % 2]

    T.evaluate(
        T.ptx_mma_sp(
            "m16n8k32",
            "row",
            "col",
            "fp16",
            "fp16",
            "fp16",
            multi_a.data,
            0,
            multi_b.data,
            0,
            accum.data,
            0,
            meta_local.data,
            0,
            0,
            False,
            dtype="float16",
        )
    )

    for i in range(4):
        C[i // 2 * 8 + tx // 4, tx % 4 * 2 + i % 2] = accum[i]


@T.prim_func
def mma_sp_m16n8k32_f16f16f32(a: T.handle, b: T.handle, c: T.handle, _metadata: T.handle):
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16], dtype="float16")
    B = T.match_buffer(b, [32, 8], dtype="float16")
    C = T.match_buffer(c, [16, 8], dtype="float32")
    metadata = T.match_buffer(_metadata, [16], dtype="uint32")
    brow = T.env_thread("blockIdx.y")
    bcol = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(brow, 1)
    T.launch_thread(bcol, 1)
    T.launch_thread(tx, 32)
    multi_a = T.decl_buffer([8], "float16", scope="local")
    multi_b = T.decl_buffer([8], "float16", scope="local")
    accum = T.decl_buffer([4], "float32", scope="local")
    meta_local = T.decl_buffer([1], "uint32", scope="local")
    for i in range(4):
        accum[i] = T.float16(0)

    for i in range(8):
        multi_a[i] = A[(i % 4) // 2 * 8 + tx // 4, i // 4 * 8 + tx % 4 * 2 + i % 2]

    for i in range(8):
        multi_b[i] = B[i // 2 * 8 + tx % 4 * 2 + i % 2, tx // 4]

    meta_local[0] = metadata[tx // 4 * 2 + tx % 2]

    T.evaluate(
        T.ptx_mma_sp(
            "m16n8k32",
            "row",
            "col",
            "fp16",
            "fp16",
            "fp32",
            multi_a.data,
            0,
            multi_b.data,
            0,
            accum.data,
            0,
            meta_local.data,
            0,
            0,
            False,
            dtype="float32",
        )
    )

    for i in range(4):
        C[i // 2 * 8 + tx // 4, tx % 4 * 2 + i % 2] = accum[i]


@tvm.testing.requires_cuda_compute_version(8)
def test_mma_sp_m16n8k16_f16():
    def get_meta_m16n8k16_half(mask):
        assert mask.shape == (16, 4, 2)
        mask = mask.reshape(16, 8)
        ret = np.zeros((8,)).astype("uint32")

        for i in range(8):
            base = 1
            for blk in range(2):
                for j in range(8):
                    ret[i] |= int(mask[blk * 8 + i, j]) * base
                    base = base << 2
        return ret

    for out_dtype in ["float16", "float32"]:
        func = mma_sp_m16n8k16_f16f16f16 if out_dtype == "float16" else mma_sp_m16n8k16_f16f16f32
        sch = tvm.tir.Schedule(func)
        cuda_mod = tvm.build(sch.mod, target="cuda")

        A_np = np.random.uniform(-1, 1, [16, 8]).astype("float16")
        B_np = np.random.uniform(-1, 1, [16, 8]).astype("float16")
        mask = gen_2in4_mask(16, 16)
        A_dense_np = get_dense_mat_by_mask(A_np, mask)
        C_np = np.matmul(A_dense_np, B_np).astype(out_dtype)
        meta = get_meta_m16n8k16_half(mask)

        ctx = tvm.cuda()
        A_tvm = tvm.nd.array(A_np, ctx)
        B_tvm = tvm.nd.array(B_np, ctx)
        C_tvm = tvm.nd.array(np.zeros_like(C_np), ctx)
        meta_tvm = tvm.nd.array(meta, ctx)
        cuda_mod(A_tvm, B_tvm, C_tvm, meta_tvm)

        tvm.testing.assert_allclose(C_tvm.numpy(), C_np, atol=1e-3, rtol=1e-3)


@tvm.testing.requires_cuda_compute_version(8)
def test_mma_sp_m16n8k32_f16():
    def get_meta_m16n8k32_half(mask):
        assert mask.shape == (16, 8, 2)
        mask = mask.reshape(16, 2, 8)
        ret = np.zeros((8, 2)).astype("uint32")

        for i in range(8):
            for k in range(2):
                base = 1
                for blk in range(2):
                    for j in range(8):
                        ret[i, k] |= int(mask[blk * 8 + i, k, j]) * base
                        base = base << 2

        return ret.reshape(16)

    for out_dtype in ["float16", "float32"]:
        func = mma_sp_m16n8k32_f16f16f16 if out_dtype == "float16" else mma_sp_m16n8k32_f16f16f32
        sch = tvm.tir.Schedule(func)
        cuda_mod = tvm.build(sch.mod, target="cuda")

        A_np = np.random.uniform(-1, 1, [16, 16]).astype("float16")
        B_np = np.random.uniform(-1, 1, [32, 8]).astype("float16")
        mask = gen_2in4_mask(16, 32)
        A_dense_np = get_dense_mat_by_mask(A_np, mask)
        C_np = np.matmul(A_dense_np, B_np).astype(out_dtype)
        meta = get_meta_m16n8k32_half(mask)

        ctx = tvm.cuda()
        A_tvm = tvm.nd.array(A_np, ctx)
        B_tvm = tvm.nd.array(B_np, ctx)
        C_tvm = tvm.nd.array(np.zeros_like(C_np), ctx)
        meta_tvm = tvm.nd.array(meta, ctx)
        cuda_mod(A_tvm, B_tvm, C_tvm, meta_tvm)

    tvm.testing.assert_allclose(C_tvm.numpy(), C_np, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_mma_sp_m16n8k16_f16()
    test_mma_sp_m16n8k32_f16()
