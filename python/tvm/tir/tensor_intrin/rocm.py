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
# pylint: disable=invalid-name,missing-function-docstring
"""Intrinsics for AMDGPU tensorization."""
from tvm.script import tir as T

from tvm.runtime import convert
from tvm.tir.expr import Cast, IntImm
from .dot_product_common import get_dp4a_intrin
from .. import TensorIntrin


lift = convert


@T.prim_func
def sdot4(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])

        C[0] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.amdgcn.sdot4"),
            T.uint32(4),
            T.reinterpret(A.vload([0], "int8x4"), dtype="int32"),
            T.reinterpret(B.vload([0], "int8x4"), dtype="int32"),
            T.int32(0),
            T.bool(1),
            dtype="int32",
        )


AMDGPU_SDOT4_INTRIN = "sdot4"

dp4a_desc, _ = get_dp4a_intrin("int8", "int8", "int32")
TensorIntrin.register(AMDGPU_SDOT4_INTRIN, dp4a_desc, sdot4)

WARP_SIZE = 64
M_DIM = 16
N_DIM = 16


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = j * 16 + i
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16 + local_id
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = i * 16 + j
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = j % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_C(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def get_mma_fill_intrin(dtype, local_size):
    zero = IntImm("int32", 0).astype(dtype)

    # Assume M = N = 16
    index_map = shared_16x16_to_local_64x4_layout_C

    @T.prim_func
    def mma_fill_desc(a: T.handle) -> None:
        C_warp = T.match_buffer(a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:WARP_SIZE, 0:local_size])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with T.block("C_warp"):
                    i, j = T.axis.remap("SS", [i0, i1])
                    thread_id, local_id = T.meta_var(index_map(i, j))
                    T.reads()
                    T.writes(C_warp[thread_id, local_id])
                    C_warp[thread_id, local_id] = zero

    @T.prim_func
    def mma_fill_impl(a: T.handle) -> None:
        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp", offset_factor=1
        )

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:WARP_SIZE, 0:local_size])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)
            for local_id in T.serial(0, local_size):
                C_warp[tx, local_id] = zero

    return mma_fill_desc, mma_fill_impl


def get_mfma_load_intrin(
    k_dim=4,
    dtype="float32",
    scope="shared",
    is_b=False,
    transposed=False,
):
    local_size = (M_DIM * k_dim) // WARP_SIZE if not is_b else (N_DIM * k_dim) // WARP_SIZE
    memory_shape = (M_DIM, k_dim)
    if is_b:
        memory_shape = (N_DIM, k_dim) if transposed else (k_dim, N_DIM)

    row_dim, col_dim = memory_shape

    if k_dim == 4:
        index_map = shared_16x4_to_local_64x1_layout_A
        reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A
        if is_b:
            index_map = (
                shared_16x4_to_local_64x1_layout_A
                if transposed
                else shared_4x16_to_local_64x1_layout_B
            )
            reverse_index_map = (
                thread_id_shared_access_64x1_to_16x4_layout_A
                if transposed
                else thread_id_shared_access_64x1_to_4x16_layout_B
            )
    elif k_dim == 16:
        index_map = shared_16x16_to_local_64x4_layout_A
        reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_A

        if is_b:
            index_map = (
                shared_16x16_to_local_64x4_layout_A
                if transposed
                else shared_16x16_to_local_64x4_layout_B
            )
            reverse_index_map = (
                thread_id_shared_access_64x4_to_16x16_layout_A
                if transposed
                else thread_id_shared_access_64x4_to_16x16_layout_B
            )
    else:
        raise ValueError("k_dim must be 4 or 16 currently")

    @T.prim_func
    def mfma_load_desc(reg_handle: T.handle, memory_handle: T.handle) -> None:
        memory = T.match_buffer(
            memory_handle,
            memory_shape,
            dtype,
            offset_factor=1,
            scope=scope,
        )
        reg = T.match_buffer(
            reg_handle, (WARP_SIZE, local_size), dtype, offset_factor=1, scope="warp"
        )

        with T.block("root"):
            T.reads(memory[0:row_dim, 0:col_dim])
            T.writes(reg[0:WARP_SIZE, 0:local_size])

            for ax0, ax1 in T.grid(row_dim, col_dim):
                with T.block("memory_reg"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(memory[v0, v1])

                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.writes(reg[thread_id, local_id])
                    reg[thread_id, local_id] = memory[v0, v1]

    @T.prim_func
    def mfma_load_impl(reg_handle: T.handle, memory_handle: T.handle) -> None:
        s0 = T.int32()
        s1 = T.int32()

        memory = T.match_buffer(
            memory_handle,
            memory_shape,
            dtype,
            align=64,
            offset_factor=1,
            scope=scope,
            strides=[s0, s1],
        )
        reg = T.match_buffer(
            reg_handle, (WARP_SIZE, local_size), dtype, align=64, offset_factor=1, scope="warp"
        )

        with T.block("root"):
            T.reads(memory[0:row_dim, 0:col_dim])
            T.writes(reg[0:WARP_SIZE, 0:local_size])
            tx = T.env_thread("threadIdx.x")
            for local_id in T.serial(0, local_size):
                row, col = T.meta_var(reverse_index_map(tx, local_id))
                T.launch_thread(tx, WARP_SIZE)
                reg[tx, local_id] = memory[row, col]

    return mfma_load_desc, mfma_load_impl


def get_mfma_intrin(k_dim, in_dtype="float32", out_dtype="float32", b_transposed=False):
    local_size = (M_DIM * k_dim) // WARP_SIZE
    local_size_out = (M_DIM * N_DIM) // WARP_SIZE
    if k_dim == 4:
        index_map_A = shared_16x4_to_local_64x1_layout_A
        index_map_B = shared_4x16_to_local_64x1_layout_B
        index_map_C = shared_16x16_to_local_64x4_layout_C
    elif k_dim == 16:
        index_map_A = shared_16x16_to_local_64x4_layout_A
        index_map_B = shared_16x16_to_local_64x4_layout_B
        index_map_C = shared_16x16_to_local_64x4_layout_C
    else:
        raise ValueError("k_dim must be 4 or 16 currently")

    out_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[out_dtype]

    in_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[in_dtype]

    mfma_intrin = f"llvm.amdgcn.mfma.{out_dtype_abbrv}.{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def maybe_cast(v):
        if out_dtype != in_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    @T.prim_func
    def mfma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        B = T.match_buffer(b, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        C = T.match_buffer(c, (WARP_SIZE, local_size_out), out_dtype, offset_factor=1, scope="warp")

        with T.block("root"):
            T.reads(
                C[0:WARP_SIZE, 0:local_size_out],
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])

            for i, j, k in T.grid(M_DIM, N_DIM, k_dim):
                with T.block("C"):
                    i, j, k = T.axis.remap("SSR", [i, j, k])
                    b_row_ind, b_col_ind = T.meta_var(maybe_swap(k, j))

                    thread_id_C, local_id_C = T.meta_var(index_map_C(i, j))
                    thread_id_A, local_id_A = T.meta_var(index_map_A(i, k))
                    thread_id_B, local_id_B = T.meta_var(index_map_B(b_row_ind, b_col_ind))

                    T.reads(
                        C[thread_id_C, local_id_C],
                        A[thread_id_A, local_id_A],
                        B[thread_id_B, local_id_B],
                    )
                    T.writes(C[thread_id_C, local_id_C])

                    C[thread_id_C, local_id_C] += maybe_cast(
                        A[thread_id_A, local_id_A]
                    ) * maybe_cast(B[thread_id_B, local_id_B])

    @T.prim_func
    def mfma_sync_impl_float(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        B = T.match_buffer(b, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        C = T.match_buffer(c, (WARP_SIZE, local_size_out), out_dtype, offset_factor=1, scope="warp")

        with T.block("root"):
            T.reads(
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
                C[0:WARP_SIZE, 0:local_size_out],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)
            C[tx, 0:local_size_out] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id(mfma_intrin),
                T.uint32(6),
                A[tx, 0:local_size],
                B[tx, 0:local_size],
                C[tx, 0:local_size_out],
                T.int32(0),
                T.int32(0),
                T.int32(0),
                dtype=f"{out_dtype}x4",
            )

    @T.prim_func
    def mfma_sync_impl_integer(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        B = T.match_buffer(b, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        C = T.match_buffer(c, (WARP_SIZE, local_size_out), out_dtype, offset_factor=1, scope="warp")

        with T.block("root"):
            T.reads(
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
                C[0:WARP_SIZE, 0:local_size_out],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            C[tx, 0:local_size_out] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id(mfma_intrin),
                T.uint32(6),
                T.call_intrin("int32", "tir.reinterpret", A[tx, 0:local_size]),
                T.call_intrin("int32", "tir.reinterpret", A[tx, 0:local_size]),
                C[tx, 0:local_size_out],
                T.int32(0),
                T.int32(0),
                T.int32(0),
                dtype=f"{out_dtype}x4",
            )

    return (
        (mfma_sync_desc, mfma_sync_impl_integer)
        if in_dtype == "int8"
        else (mfma_sync_desc, mfma_sync_impl_float)
    )


def get_mfma_store_intrin(local_size=4, dtype="float32", scope="global"):
    index_map = shared_16x16_to_local_64x4_layout_C

    @T.prim_func
    def mfma_store_desc(a: T.handle, c: T.handle) -> None:
        C_warp = T.match_buffer(a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")
        C = T.match_buffer(c, [M_DIM, N_DIM], dtype=dtype, scope=scope)

        with T.block("root"):
            T.reads(C_warp[0:WARP_SIZE, 0:local_size])
            T.writes(C[0:M_DIM, 0:N_DIM])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with T.block("C_warp"):
                    v0, v1 = T.axis.remap("SS", [i0, i1])
                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.reads(C_warp[thread_id, local_id])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_warp[thread_id, local_id]

    @T.prim_func
    def mfma_store_impl(a: T.handle, c: T.handle) -> None:
        s0 = T.int32()
        s1 = T.int32()

        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp", offset_factor=1
        )
        C = T.match_buffer(
            c, [M_DIM, N_DIM], dtype=dtype, scope=scope, offset_factor=1, strides=[s0, s1]
        )

        with T.block("root"):
            T.reads(C_warp[0:WARP_SIZE, 0:local_size])
            T.writes(C[0:M_DIM, 0:N_DIM])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)
            for i in range(local_size):
                C[((tx // 16) * 4) + i, (tx % 16)] = C_warp[tx, i]

    return mfma_store_desc, mfma_store_impl


ROCM_MFMA_fill_16x16_f32_INTRIN = "ROCM_mfma_fill_16x16_f32"
TensorIntrin.register(ROCM_MFMA_fill_16x16_f32_INTRIN, *get_mma_fill_intrin("float32", 4))

ROCM_MFMA_fill_16x16_i32_INTRIN = "ROCM_mfma_fill_16x16_i32"
TensorIntrin.register(ROCM_MFMA_fill_16x16_i32_INTRIN, *get_mma_fill_intrin("int", 4))

ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN = "rocm_mfma_load_16x16_a_shared_s8"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN, *get_mfma_load_intrin(16, "int8", "shared")
)
ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN = "rocm_mfma_load_b_16x16_shared_s8"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN, *get_mfma_load_intrin(16, "int8", "shared", is_b=True)
)

ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN = "rocm_mfma_load_16x16_a_shared_f16"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN, *get_mfma_load_intrin(16, "float16", "shared")
)
ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN = "rocm_mfma_load_b_16x16_shared_f16"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
    *get_mfma_load_intrin(16, "float16", "shared", is_b=True),
)

ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN = "rocm_mfma_load_16x4_a_shared_f32"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN, *get_mfma_load_intrin(4, "float32", "shared")
)
ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN = "rocm_mfma_load_b_16x4_shared_f32"
TensorIntrin.register(
    ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
    *get_mfma_load_intrin(4, "float32", "shared", is_b=True),
)


ROCM_MFMA_f32f32f32_INTRIN = "rocm_mfma_f32f32f32"
TensorIntrin.register(ROCM_MFMA_f32f32f32_INTRIN, *get_mfma_intrin(4, "float32", "float32"))

ROCM_MFMA_f16f16f32_INTRIN = "rocm_mfma_f16f16f32"
TensorIntrin.register(ROCM_MFMA_f16f16f32_INTRIN, *get_mfma_intrin(16, "float16", "float32"))

ROCM_MFMA_s8s8s32_INTRIN = "rocm_mfma_s8s8s32"
TensorIntrin.register(ROCM_MFMA_s8s8s32_INTRIN, *get_mfma_intrin(16, "int8", "int32"))

ROCM_MFMA_STORE_16x16_s32_INTRIN = "rocm_mfma_store_16x16_s32"
TensorIntrin.register(
    ROCM_MFMA_STORE_16x16_s32_INTRIN, *get_mfma_store_intrin(4, "int32", "global")
)

ROCM_MFMA_STORE_16x16_f32_INTRIN = "rocm_mfma_store_16x16_f32"
TensorIntrin.register(
    ROCM_MFMA_STORE_16x16_f32_INTRIN, *get_mfma_store_intrin(4, "float32", "global")
)
