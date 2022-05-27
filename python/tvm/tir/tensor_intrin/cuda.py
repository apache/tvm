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
"""Intrinsics for tensorization on NVIDIA GPU."""
from tvm.script import tir as T
from .. import IntImm, Cast
from ..._ffi import register_func
from ...runtime import convert
from .. import TensorIntrin


def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_16x32_to_ldmatrix_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


def shared_32x16_to_ldmatrix_32x16_layout(i, j):
    thread_id = (i % 4) + 4 * (j % 8)
    return thread_id, 8 * (j // 8) + (i // 16) * 4 + i % 4


@register_func("tir.index_map.shared_16x16_to_ldmatrix_32x8_layout")
def index_map_shared_16x16_to_ldmatrix_32x8_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_ldmatrix_32x8_layout(i, j)
    return convert([thread_id, local_id])


lift = convert

M_DIM = 16
N_DIM = 16
WARP_SIZE = 32
HALF_WARP = WARP_SIZE // 2
HALF_WARP_expr = lift(HALF_WARP)


def get_ldmatrix_intrin(k_dim, dtype, is_b, transposed):
    local_size = (M_DIM * k_dim) // WARP_SIZE
    shared_offset = None
    index_map = None

    if transposed:
        assert is_b, "Transposed A matrix not supported"

    ldmatrix_col_major = is_b and not transposed

    if k_dim == 16:
        assert dtype == "float16"

        index_map = shared_16x16_to_ldmatrix_32x8_layout

        if transposed:
            shared_offset = (
                lambda tx, stride: stride * 8 * (tx // HALF_WARP_expr)
                + stride * (tx % 8)
                + 8 * ((tx % HALF_WARP_expr) // 8)
            )
        else:
            shared_offset = lambda tx, stride: stride * (tx % HALF_WARP_expr) + 8 * (
                tx // HALF_WARP_expr
            )
    else:
        assert (
            k_dim == 32 and dtype == "int8"
        ), "Only k_dim == 16 (float16) or k_dim == 32 (int8) supported for now"

        if ldmatrix_col_major:
            index_map = shared_32x16_to_ldmatrix_32x16_layout
            # A dummy offset, ldmatrix cannot be used for int8 + trans case.
            # We still use the ldmatrix intrinsic, but lower it to a manual loop in the codegen.
            # Only the stride information is required.
            shared_offset = lambda _, stride: stride
        elif is_b and transposed:
            index_map = shared_16x32_to_ldmatrix_32x16_layout
            shared_offset = (
                lambda tx, stride: stride * 8 * (tx // HALF_WARP_expr)
                + (tx % 8) * stride
                + 16 * ((tx % HALF_WARP_expr) // 8)
            )
        else:
            index_map = shared_16x32_to_ldmatrix_32x16_layout
            shared_offset = lambda tx, stride: stride * (tx % 16) + 16 * (tx // 16)

    assert index_map and shared_offset

    if is_b and not transposed:
        row_dim = k_dim
        col_dim = M_DIM
    else:
        row_dim = M_DIM
        col_dim = k_dim

    shmem_shape = (row_dim, col_dim)

    @T.prim_func
    def ldmatrix_desc(warp_handle: T.handle, shared_handle: T.handle) -> None:
        shared = T.match_buffer(
            shared_handle, shmem_shape, dtype, align=128, offset_factor=16, scope="shared"
        )
        warp = T.match_buffer(
            warp_handle, (WARP_SIZE, local_size), dtype, align=128, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(shared[0:row_dim, 0:col_dim])
            T.writes(warp[0:WARP_SIZE, 0:local_size])

            for ax0, ax1 in T.grid(row_dim, col_dim):
                with T.block("shared_warp"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(shared[v0, v1])

                    thread_id, local_id = index_map(v0, v1)
                    T.writes(warp[thread_id, local_id])
                    warp[thread_id, local_id] = shared[v0, v1]

    @T.prim_func
    def ldmatrix_impl(warp_handle: T.handle, shared_handle: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        shared = T.match_buffer(
            shared_handle,
            shmem_shape,
            dtype,
            align=128,
            offset_factor=16,
            scope="shared",
            strides=[s0, s1],
        )
        warp = T.match_buffer(
            warp_handle, (WARP_SIZE, local_size), dtype, align=128, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(shared[0:row_dim, 0:col_dim])
            T.writes(warp[0:WARP_SIZE, 0:local_size])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(
                T.ptx_ldmatrix(
                    ldmatrix_col_major,
                    4,  # Always load 4 matrices
                    ".b16",
                    warp.data,
                    warp.elem_offset + lift(local_size) * tx,
                    shared.access_ptr("r"),
                    shared_offset(tx, s0),
                    dtype=dtype,
                )
            )

    return ldmatrix_desc, ldmatrix_impl


def get_mma_intrin(k_dim, out_dtype, b_transposed):
    local_size = (M_DIM * k_dim) // WARP_SIZE
    local_size_out = (M_DIM * N_DIM) // 32

    index_map_C = shared_16x16_to_ldmatrix_32x8_layout

    if k_dim == 16:
        index_map_A = shared_16x16_to_ldmatrix_32x8_layout
        index_map_B = shared_16x16_to_ldmatrix_32x8_layout
        mma_prefix = "m16n8k16"
    elif k_dim == 32 and b_transposed:
        index_map_A = index_map_B = shared_16x32_to_ldmatrix_32x16_layout
        mma_prefix = "m16n8k32"
    elif k_dim == 32 and not b_transposed:
        index_map_A = shared_16x32_to_ldmatrix_32x16_layout
        index_map_B = shared_32x16_to_ldmatrix_32x16_layout
        mma_prefix = "m16n8k32"
    else:
        assert False

    out_dtype_abbrv = {"float16": "fp16", "float32": "fp32", "int32": "int32"}[out_dtype]

    if out_dtype in ["float16", "float32"]:
        in_dtype = "float16"
        in_dtype_abbrv = "fp16"
    else:
        in_dtype = "int8"
        in_dtype_abbrv = "int8"

    def maybe_cast(v):
        if out_dtype in ["float32", "int32"]:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    @T.prim_func
    def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (WARP_SIZE, local_size), in_dtype, align=128, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (WARP_SIZE, local_size), in_dtype, align=128, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, align=128, offset_factor=16, scope="warp"
        )

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
                    b_row_ind, b_col_ind = maybe_swap(k, j)

                    thread_id_C, local_id_C = index_map_C(i, j)
                    thread_id_A, local_id_A = index_map_A(i, k)
                    thread_id_B, local_id_B = index_map_B(b_row_ind, b_col_ind)

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
    def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (WARP_SIZE, local_size), in_dtype, align=128, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (WARP_SIZE, local_size), in_dtype, align=128, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, align=128, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(
                C[0:WARP_SIZE, 0:local_size_out],
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(
                T.ptx_mma(
                    mma_prefix,
                    "row",
                    "col",
                    in_dtype_abbrv,
                    in_dtype_abbrv,
                    out_dtype_abbrv,
                    A.data,
                    A.elem_offset + tx * lift(local_size),
                    B.data,
                    B.elem_offset + tx * lift(local_size),
                    C.data,
                    C.elem_offset + tx * lift(local_size_out),
                    False,
                    dtype=out_dtype,
                )
            )

            T.evaluate(
                T.ptx_mma(
                    mma_prefix,
                    "row",
                    "col",
                    in_dtype_abbrv,
                    in_dtype_abbrv,
                    out_dtype_abbrv,
                    A.data,
                    A.elem_offset + tx * lift(local_size),
                    B.data,
                    B.elem_offset + tx * lift(local_size) + lift(local_size) // 2,
                    C.data,
                    C.elem_offset + tx * lift(local_size_out) + lift(local_size_out) // 2,
                    False,
                    dtype=out_dtype,
                )
            )

    return mma_sync_desc, mma_sync_impl


def get_mma_fill_intrin(dtype, local_size):
    zero = IntImm("int32", 0).astype(dtype)

    # Assume M = N = 16
    index_map = shared_16x16_to_ldmatrix_32x8_layout

    @T.prim_func
    def mma_fill_desc(a: T.handle) -> None:
        C_warp = T.match_buffer(a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:WARP_SIZE, 0:local_size])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with T.block("C_warp"):
                    i, j = T.axis.remap("SS", [i0, i1])
                    thread_id, local_id = index_map(i, j)
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

            T.evaluate(T.mma_fill(local_size, C_warp.data, C_warp.elem_offset, dtype=dtype))

    return mma_fill_desc, mma_fill_impl


def get_mma_store_intrin(dtype, local_size, scope="global"):
    # Assume M = N = 16
    index_map = shared_16x16_to_ldmatrix_32x8_layout

    @T.prim_func
    def mma_store_desc(a: T.handle, c: T.handle) -> None:
        C_warp = T.match_buffer(a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")
        C = T.match_buffer(c, [M_DIM, N_DIM], dtype=dtype, scope=scope)

        with T.block("root"):
            T.reads(C_warp[0:WARP_SIZE, 0:local_size])
            T.writes(C[0:M_DIM, 0:N_DIM])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with T.block("C_warp"):
                    v0, v1 = T.axis.remap("SS", [i0, i1])
                    thread_id, local_id = index_map(v0, v1)
                    T.reads(C_warp[thread_id, local_id])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_warp[thread_id, local_id]

    @T.prim_func
    def mma_store_impl(a: T.handle, c: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")

        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp", offset_factor=1
        )
        C = T.match_buffer(
            c, [M_DIM, N_DIM], dtype=dtype, scope="global", offset_factor=1, strides=[s0, s1]
        )

        with T.block("root"):
            T.reads(C_warp[0:WARP_SIZE, 0:local_size])
            T.writes(C[0:M_DIM, 0:N_DIM])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(
                T.mma_store(
                    M_DIM,
                    N_DIM,
                    C.access_ptr("w"),
                    C_warp.data,
                    C_warp.elem_offset,
                    s0,
                    dtype=dtype,
                )
            )

    return mma_store_desc, mma_store_impl


LDMATRIX_16x16_A_INTRIN = "mma.ldmatrix_16x16_a"
TensorIntrin.register(LDMATRIX_16x16_A_INTRIN, *get_ldmatrix_intrin(16, "float16", False, False))

LDMATRIX_16x16_B_INTRIN = "mma.ldmatrix_16x16_b"
TensorIntrin.register(LDMATRIX_16x16_B_INTRIN, *get_ldmatrix_intrin(16, "float16", True, False))

LDMATRIX_16x16_B_TRANS_INTRIN = "mma.ldmatrix_16x16_b_trans"
TensorIntrin.register(
    LDMATRIX_16x16_B_TRANS_INTRIN, *get_ldmatrix_intrin(16, "float16", True, True)
)

LDMATRIX_16x32_A_INTRIN = "mma.ldmatrix_16x32_a"
TensorIntrin.register(LDMATRIX_16x32_A_INTRIN, *get_ldmatrix_intrin(32, "int8", False, False))

LDMATRIX_32x16_B_INTRIN = "mma.ldmatrix_32x16_b"
TensorIntrin.register(LDMATRIX_32x16_B_INTRIN, *get_ldmatrix_intrin(32, "int8", True, False))

LDMATRIX_16x32_B_TRANS_INTRIN = "mma.ldmatrix_16x32_b_trans"
TensorIntrin.register(LDMATRIX_16x32_B_TRANS_INTRIN, *get_ldmatrix_intrin(32, "int8", True, True))

MMA_f16f16f32_INTRIN = "mma_f16f16f32"
TensorIntrin.register(MMA_f16f16f32_INTRIN, *get_mma_intrin(16, "float32", False))

MMA_f16f16f32_TRANS_INTRIN = "mma_f16f16f32_trans"
TensorIntrin.register(MMA_f16f16f32_TRANS_INTRIN, *get_mma_intrin(16, "float32", True))

MMA_f16f16f16_INTRIN = "mma_f16f16f16"
TensorIntrin.register(MMA_f16f16f16_INTRIN, *get_mma_intrin(16, "float16", False))

MMA_f16f16f16_TRANS_INTRIN = "mma_f16f16f16_trans"
TensorIntrin.register(MMA_f16f16f16_TRANS_INTRIN, *get_mma_intrin(16, "float16", True))

MMA_i8i8i32_INTRIN = "mma_i8i8i32"
TensorIntrin.register(MMA_i8i8i32_INTRIN, *get_mma_intrin(32, "int32", False))

MMA_i8i8i32_TRANS_INTRIN = "mma_i8i8i32_trans"
TensorIntrin.register(MMA_i8i8i32_TRANS_INTRIN, *get_mma_intrin(32, "int32", True))

MMA_fill_16x16_f32_INTRIN = "mma_fill_16x16_f32"
TensorIntrin.register(MMA_fill_16x16_f32_INTRIN, *get_mma_fill_intrin("float32", 8))

MMA_fill_16x16_f16_INTRIN = "mma_fill_16x16_f16"
TensorIntrin.register(MMA_fill_16x16_f16_INTRIN, *get_mma_fill_intrin("float16", 8))

MMA_fill_16x16_i32_INTRIN = "mma_fill_16x16_i32"
TensorIntrin.register(MMA_fill_16x16_i32_INTRIN, *get_mma_fill_intrin("int32", 8))

MMA_store_16x16_f32_global_INTRIN = "mma_store_16x16_f32_global_"
TensorIntrin.register(
    MMA_store_16x16_f32_global_INTRIN, *get_mma_store_intrin("float32", 8, "global")
)

MMA_store_16x16_f16_global_INTRIN = "mma_store_16x16_f16_global_"
TensorIntrin.register(
    MMA_store_16x16_f16_global_INTRIN, *get_mma_store_intrin("float16", 8, "global")
)

MMA_store_16x16_i32_global_INTRIN = "mma_store_16x16_i32_global_"
TensorIntrin.register(
    MMA_store_16x16_i32_global_INTRIN, *get_mma_store_intrin("int32", 8, "global")
)
