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
# pylint: disable=invalid-name,missing-function-docstring,unused-variable
"""Intrinsics for tensorization on NVIDIA GPU."""
from typing import Dict, Optional, Tuple, Literal

from tvm._ffi import register_func
from tvm.runtime import convert
from tvm.script import tir as T
from tvm.tir.function import PrimFunc
from tvm.tir import Cast, IntImm, TensorIntrin


def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_16x32_to_ldmatrix_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


def shared_32x16_to_ldmatrix_32x16_layout(i, j):
    thread_id = (i % 16) // 4 + 4 * (j % 8)
    return thread_id, 8 * (j // 8) + (i // 16) * 4 + i % 4


def ldmatrix_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col


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


def get_ldmatrix_intrin(
    k_dim: int,
    dtype: str,
    matrix_name: Literal["A", "B"],
    transposed: bool,
    shared_scope: str = "shared",
):
    local_size = (M_DIM * k_dim) // WARP_SIZE
    smem_offset = None
    index_map = None

    if matrix_name == "A":
        transpose_in_ldmatrix = transposed
        # transpose_layout_for_ldmatrix_input: Every thread loads 8 bytes data. This determines
        # which 8 bytes every thread loads.
        # If transpose_layout_for_ldmatrix_input is False, the load pattern is
        # T0  T0  T0  T0  T16 T16 T16 T16
        # T1  T1  T1  T1  T17 T17 T17 T17
        # ...
        # T8  T8  T8  T8  T24 T24 T24 T24
        # T9  T9  T9  T9  T25 T25 T25 T25
        # ...
        # T15 T15 T15 T15 T31 T31 T31 T31
        # Otherwise, the load pattern is
        # T0  T0  T0  T0  T8  T8  T8  T8
        # T1  T1  T1  T1  T9  T9  T9  T9
        # ...
        # T7  T7  T7  T7  T15 T15 T15 T15
        # T16 T16 T16 T16 T24 T24 T24 T24
        # T17 T17 T17 T17 T25 T25 T25 T25
        # ...
        # T23 T23 T23 T23 T31 T31 T31 T31
        transpose_layout_for_ldmatrix_input = transposed
        smem_tile_row, smem_tile_col = (M_DIM, k_dim) if not transposed else (k_dim, M_DIM)
    else:
        assert matrix_name == "B"
        transpose_in_ldmatrix = not transposed
        transpose_layout_for_ldmatrix_input = transposed
        smem_tile_row, smem_tile_col = (k_dim, N_DIM) if not transposed else (N_DIM, k_dim)

    if k_dim == 16:
        assert dtype == "float16"

        index_map = shared_16x16_to_ldmatrix_32x8_layout

        if transpose_layout_for_ldmatrix_input:
            smem_offset = (
                lambda tx, stride: stride * 8 * (tx // HALF_WARP_expr)
                + stride * (tx % 8)
                + 8 * ((tx % HALF_WARP_expr) // 8)
            )
        else:
            smem_offset = lambda tx, stride: stride * (tx % HALF_WARP_expr) + 8 * (
                tx // HALF_WARP_expr
            )
    else:
        # TODO(yixin): Support TN and TT matmul for int8
        assert (
            matrix_name == "B" or not transposed
        ), "Now only B matrix can be transposed for int8 matmul"
        assert k_dim == 32 and (
            dtype == "int8" or dtype == "e4m3_float8" or dtype == "e5m2_float8"
        ), "Only k_dim == 16 (float16) or k_dim == 32 (int8) supported for now"

        if matrix_name == "B" and not transposed:
            index_map = shared_32x16_to_ldmatrix_32x16_layout
            # A dummy offset, ldmatrix cannot be used for int8 + trans case.
            # We still use the ldmatrix intrinsic, but lower it to a manual loop in the codegen.
            # Only the stride information is required.
            smem_offset = lambda _, stride: stride
        elif matrix_name == "B" and transposed:
            index_map = shared_16x32_to_ldmatrix_32x16_layout
            smem_offset = (
                lambda tx, stride: stride * 8 * (tx // HALF_WARP_expr)
                + (tx % 8) * stride
                + 16 * ((tx % HALF_WARP_expr) // 8)
            )
        else:  # A, not transposed
            index_map = shared_16x32_to_ldmatrix_32x16_layout
            smem_offset = lambda tx, stride: stride * (tx % 16) + 16 * (tx // 16)

    offset_factor = smem_tile_col

    @T.prim_func
    def ldmatrix_desc(warp_handle: T.handle, shared_handle: T.handle) -> None:
        shared = T.match_buffer(
            shared_handle,
            (smem_tile_row, smem_tile_col),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=shared_scope,
        )
        warp = T.match_buffer(
            warp_handle,
            (WARP_SIZE, local_size),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="warp",
        )

        with T.block("root"):
            T.reads(shared[0:smem_tile_row, 0:smem_tile_col])
            T.writes(warp[0:WARP_SIZE, 0:local_size])

            for ax0, ax1 in T.grid(smem_tile_row, smem_tile_col):
                with T.block("shared_warp"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(shared[v0, v1])

                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.writes(warp[thread_id, local_id])
                    warp[thread_id, local_id] = shared[v0, v1]

    @T.prim_func
    def ldmatrix_impl(warp_handle: T.handle, shared_handle: T.handle) -> None:
        s0 = T.int32()
        s1 = T.int32()
        shared = T.match_buffer(
            shared_handle,
            (smem_tile_row, smem_tile_col),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=shared_scope,
            strides=[s0, s1],
        )
        warp = T.match_buffer(
            warp_handle,
            (WARP_SIZE, local_size),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="warp",
        )

        with T.block("root"):
            T.reads(shared[0:smem_tile_row, 0:smem_tile_col])
            T.writes(warp[0:WARP_SIZE, 0:local_size])
            for tx in T.thread_binding(0, WARP_SIZE, "threadIdx.x"):
                T.evaluate(
                    T.ptx_ldmatrix(
                        transpose_in_ldmatrix,
                        4,  # Always load 4 matrices
                        ".b16",
                        warp.data,
                        warp.elem_offset + lift(local_size) * tx,
                        shared.access_ptr("r"),
                        smem_offset(tx, s0),
                        dtype=dtype,
                    )
                )

    return ldmatrix_desc, ldmatrix_impl


LDMATRIX_f16_A_INTRIN = "mma_ldmatrix_f16_a"
TensorIntrin.register(LDMATRIX_f16_A_INTRIN, *get_ldmatrix_intrin(16, "float16", "A", False))

LDMATRIX_f16_B_INTRIN = "mma_ldmatrix_f16_b"
TensorIntrin.register(LDMATRIX_f16_B_INTRIN, *get_ldmatrix_intrin(16, "float16", "B", False))

LDMATRIX_f16_A_TRANS_INTRIN = "mma_ldmatrix_f16_a_trans"
TensorIntrin.register(LDMATRIX_f16_A_TRANS_INTRIN, *get_ldmatrix_intrin(16, "float16", "A", True))

LDMATRIX_f16_B_TRANS_INTRIN = "mma_ldmatrix_f16_b_trans"
TensorIntrin.register(LDMATRIX_f16_B_TRANS_INTRIN, *get_ldmatrix_intrin(16, "float16", "B", True))

LDMATRIX_f16_A_DYN_INTRIN = "mma_ldmatrix_f16_a_dyn"
TensorIntrin.register(
    LDMATRIX_f16_A_DYN_INTRIN, *get_ldmatrix_intrin(16, "float16", "A", False, "shared.dyn")
)

LDMATRIX_f16_B_DYN_INTRIN = "mma_ldmatrix_f16_b_dyn"
TensorIntrin.register(
    LDMATRIX_f16_B_DYN_INTRIN, *get_ldmatrix_intrin(16, "float16", "B", False, "shared.dyn")
)

LDMATRIX_f16_A_TRANS_DYN_INTRIN = "mma_ldmatrix_f16_a_trans_dyn"
TensorIntrin.register(
    LDMATRIX_f16_A_TRANS_DYN_INTRIN, *get_ldmatrix_intrin(16, "float16", "A", True, "shared.dyn")
)

LDMATRIX_f16_B_TRANS_DYN_INTRIN = "mma_ldmatrix_f16_b_trans_dyn"
TensorIntrin.register(
    LDMATRIX_f16_B_TRANS_DYN_INTRIN, *get_ldmatrix_intrin(16, "float16", "B", True, "shared.dyn")
)

LDMATRIX_i8_A_INTRIN = "mma_ldmatrix_i8_a"
TensorIntrin.register(LDMATRIX_i8_A_INTRIN, *get_ldmatrix_intrin(32, "int8", "A", False))

LDMATRIX_i8_B_INTRIN = "mma_ldmatrix_i8_b"
TensorIntrin.register(LDMATRIX_i8_B_INTRIN, *get_ldmatrix_intrin(32, "int8", "B", False))

LDMATRIX_i8_B_TRANS_INTRIN = "mma_ldmatrix_i8_b_trans"
TensorIntrin.register(LDMATRIX_i8_B_TRANS_INTRIN, *get_ldmatrix_intrin(32, "int8", "B", True))

LDMATRIX_e4m3_A_INTRIN = "mma_ldmatrix_e4m3_a"
TensorIntrin.register(LDMATRIX_e4m3_A_INTRIN, *get_ldmatrix_intrin(32, "e4m3_float8", "A", False))

LDMATRIX_e4m3_B_INTRIN = "mma_ldmatrix_e4m3_b"
TensorIntrin.register(LDMATRIX_e4m3_B_INTRIN, *get_ldmatrix_intrin(32, "e4m3_float8", "B", False))

LDMATRIX_e4m3_B_TRANS_INTRIN = "mma_ldmatrix_e4m3_b_trans"
TensorIntrin.register(
    LDMATRIX_e4m3_B_TRANS_INTRIN, *get_ldmatrix_intrin(32, "e4m3_float8", "B", True)
)

LDMATRIX_e5m2_A_INTRIN = "mma_ldmatrix_e5m2_a"
TensorIntrin.register(LDMATRIX_e5m2_A_INTRIN, *get_ldmatrix_intrin(32, "e5m2_float8", "A", False))

LDMATRIX_e5m2_B_INTRIN = "mma_ldmatrix_e5m2_b"
TensorIntrin.register(LDMATRIX_e5m2_B_INTRIN, *get_ldmatrix_intrin(32, "e5m2_float8", "B", False))

LDMATRIX_e5m2_B_TRANS_INTRIN = "mma_ldmatrix_e5m2_b_trans"
TensorIntrin.register(
    LDMATRIX_e5m2_B_TRANS_INTRIN, *get_ldmatrix_intrin(32, "e5m2_float8", "B", True)
)


def get_mma_intrin(
    k_dim,
    a_dtype="float16",
    b_dtype="float16",
    out_dtype="float16",
    a_transposed=False,
    b_transposed=False,
):
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

    dtype_abbrv = {
        "float16": "fp16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "e4m3_float8": "e4m3",
        "e5m2_float8": "e5m2",
    }
    a_dtype_abbrv = dtype_abbrv[a_dtype]
    b_dtype_abbrv = dtype_abbrv[b_dtype]
    out_dtype_abbrv = dtype_abbrv[out_dtype]

    def cast_to_out_dtype(v):
        if out_dtype in ["float32", "int32"]:
            return Cast(out_dtype, v)
        return v

    def swap_if_flag(i, j, flag):
        return (j, i) if flag else (i, j)

    A_offset_factor = M_DIM if a_transposed else k_dim
    B_offset_factor = k_dim if b_transposed else N_DIM
    out_offset_factor = N_DIM

    @T.prim_func
    def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (WARP_SIZE, local_size),
            a_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="warp",
        )
        B = T.match_buffer(
            b,
            (WARP_SIZE, local_size),
            b_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="warp",
        )
        C = T.match_buffer(
            c,
            (WARP_SIZE, local_size_out),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="warp",
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
                    a_row_ind, a_col_ind = T.meta_var(swap_if_flag(i, k, a_transposed))
                    b_row_ind, b_col_ind = T.meta_var(swap_if_flag(k, j, b_transposed))

                    thread_id_C, local_id_C = T.meta_var(index_map_C(i, j))
                    thread_id_A, local_id_A = T.meta_var(index_map_A(a_row_ind, a_col_ind))
                    thread_id_B, local_id_B = T.meta_var(index_map_B(b_row_ind, b_col_ind))

                    T.reads(
                        C[thread_id_C, local_id_C],
                        A[thread_id_A, local_id_A],
                        B[thread_id_B, local_id_B],
                    )
                    T.writes(C[thread_id_C, local_id_C])

                    C[thread_id_C, local_id_C] += cast_to_out_dtype(
                        A[thread_id_A, local_id_A]
                    ) * cast_to_out_dtype(B[thread_id_B, local_id_B])

    @T.prim_func
    def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (WARP_SIZE, local_size),
            a_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="warp",
        )
        B = T.match_buffer(
            b,
            (WARP_SIZE, local_size),
            b_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="warp",
        )
        C = T.match_buffer(
            c,
            (WARP_SIZE, local_size_out),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="warp",
        )

        with T.block("root"):
            T.reads(
                C[0:WARP_SIZE, 0:local_size_out],
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])

            for tx in T.thread_binding(0, WARP_SIZE, "threadIdx.x"):
                T.evaluate(
                    T.ptx_mma(
                        mma_prefix,
                        "row",
                        "col",
                        a_dtype_abbrv,
                        b_dtype_abbrv,
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
                        a_dtype_abbrv,
                        b_dtype_abbrv,
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


MMA_f16f16f32_INTRIN = "mma_f16f16f32"
TensorIntrin.register(
    MMA_f16f16f32_INTRIN, *get_mma_intrin(16, "float16", "float16", "float32", False, False)
)

MMA_f16f16f32_TRANS_B_INTRIN = "mma_f16f16f32_trans_b"
TensorIntrin.register(
    MMA_f16f16f32_TRANS_B_INTRIN, *get_mma_intrin(16, "float16", "float16", "float32", False, True)
)

MMA_f16f16f32_TRANS_A_INTRIN = "mma_f16f16f32_trans_a"
TensorIntrin.register(
    MMA_f16f16f32_TRANS_A_INTRIN, *get_mma_intrin(16, "float16", "float16", "float32", True, False)
)

MMA_f16f16f32_TRANS_A_TRANS_B_INTRIN = "mma_f16f16f32_trans_a_trans_b"
TensorIntrin.register(
    MMA_f16f16f32_TRANS_A_TRANS_B_INTRIN,
    *get_mma_intrin(16, "float16", "float16", "float32", True, True),
)

MMA_f16f16f16_INTRIN = "mma_f16f16f16"
TensorIntrin.register(
    MMA_f16f16f16_INTRIN, *get_mma_intrin(16, "float16", "float16", "float16", False, False)
)

MMA_f16f16f16_TRANS_B_INTRIN = "mma_f16f16f16_trans_b"
TensorIntrin.register(
    MMA_f16f16f16_TRANS_B_INTRIN, *get_mma_intrin(16, "float16", "float16", "float16", False, True)
)

MMA_f16f16f16_TRANS_A_INTRIN = "mma_f16f16f16_trans_a"
TensorIntrin.register(
    MMA_f16f16f16_TRANS_A_INTRIN, *get_mma_intrin(16, "float16", "float16", "float16", True, False)
)

MMA_f16f16f16_TRANS_A_TRANS_B_INTRIN = "mma_f16f16f16_trans_a_trans_b"
TensorIntrin.register(
    MMA_f16f16f16_TRANS_A_TRANS_B_INTRIN,
    *get_mma_intrin(16, "float16", "float16", "float16", True, True),
)

MMA_i8i8i32_INTRIN = "mma_i8i8i32"
TensorIntrin.register(
    MMA_i8i8i32_INTRIN, *get_mma_intrin(32, "int8", "int8", "int32", False, False)
)

MMA_i8i8i32_TRANS_B_INTRIN = "mma_i8i8i32_trans_b"
TensorIntrin.register(
    MMA_i8i8i32_TRANS_B_INTRIN, *get_mma_intrin(32, "int8", "int8", "int32", False, True)
)

MMA_e5m2e5m2f32_INTRIN = "mma_e5m2e5m2f32"
TensorIntrin.register(
    MMA_e5m2e5m2f32_INTRIN,
    *get_mma_intrin(32, "e5m2_float8", "e5m2_float8", "float32", False, False),
)

MMA_e5m2e5m2f32_TRANS_B_INTRIN = "mma_e5m2e5m2f32_trans_b"
TensorIntrin.register(
    MMA_e5m2e5m2f32_TRANS_B_INTRIN,
    *get_mma_intrin(32, "e5m2_float8", "e5m2_float8", "float32", False, True),
)

MMA_e4m3e4m3f32_INTRIN = "mma_e4m3e4m3f32"
TensorIntrin.register(
    MMA_e4m3e4m3f32_INTRIN,
    *get_mma_intrin(32, "e4m3_float8", "e4m3_float8", "float32", False, False),
)

MMA_e4m3e4m3f32_TRANS_B_INTRIN = "mma_e4m3e4m3f32_trans_b"
TensorIntrin.register(
    MMA_e4m3e4m3f32_TRANS_B_INTRIN,
    *get_mma_intrin(32, "e4m3_float8", "e4m3_float8", "float32", False, True),
)


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

            for tx in T.thread_binding(0, WARP_SIZE, "threadIdx.x"):
                T.evaluate(T.mma_fill(local_size, C_warp.data, C_warp.elem_offset, dtype=dtype))

    return mma_fill_desc, mma_fill_impl


MMA_fill_16x16_f32_INTRIN = "mma_fill_16x16_f32"
TensorIntrin.register(MMA_fill_16x16_f32_INTRIN, *get_mma_fill_intrin("float32", 8))

MMA_fill_16x16_f16_INTRIN = "mma_fill_16x16_f16"
TensorIntrin.register(MMA_fill_16x16_f16_INTRIN, *get_mma_fill_intrin("float16", 8))

MMA_fill_16x16_i32_INTRIN = "mma_fill_16x16_i32"
TensorIntrin.register(MMA_fill_16x16_i32_INTRIN, *get_mma_fill_intrin("int32", 8))


def get_mma_store_intrin(dtype, local_size, scope="global", use_mma_store_intrinic=True):
    # Assume M = N = 16
    index_map = shared_16x16_to_ldmatrix_32x8_layout
    index_map_rev = ldmatrix_32x8_to_shared_16x16_layout

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
                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.reads(C_warp[thread_id, local_id])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_warp[thread_id, local_id]

    if use_mma_store_intrinic:

        @T.prim_func
        def mma_store_impl(a: T.handle, c: T.handle) -> None:
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

                for tx in T.thread_binding(0, WARP_SIZE, "threadIdx.x"):
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

    else:

        @T.prim_func
        def mma_store_impl(a: T.handle, c: T.handle) -> None:
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

                for tx in T.thread_binding(0, WARP_SIZE, "threadIdx.x"):
                    for local_id in T.serial(local_size):
                        row, col = T.meta_var(index_map_rev(tx, local_id))
                        C[row, col] = C_warp[tx, local_id]

    return mma_store_desc, mma_store_impl


MMA_store_16x16_f32_global_INTRIN = "mma_store_16x16_f32_global_"
TensorIntrin.register(
    MMA_store_16x16_f32_global_INTRIN, *get_mma_store_intrin("float32", 8, "global", True)
)

MMA_store_16x16_f32_shared_dyn_INTRIN = "mma_store_16x16_f32_shared_dyn_"
TensorIntrin.register(
    MMA_store_16x16_f32_shared_dyn_INTRIN, *get_mma_store_intrin("float32", 8, "shared.dyn", True)
)

MMA_store_16x16_f32_shared_dyn_INTRIN_SIMPLE = "mma_store_16x16_f32_shared_dyn_simple_"
TensorIntrin.register(
    MMA_store_16x16_f32_shared_dyn_INTRIN_SIMPLE,
    *get_mma_store_intrin("float32", 8, "shared.dyn", False),
)

MMA_store_16x16_f16_shared_dyn_INTRIN_SIMPLE = "mma_store_16x16_f16_shared_dyn_simple_"
TensorIntrin.register(
    MMA_store_16x16_f16_shared_dyn_INTRIN_SIMPLE,
    *get_mma_store_intrin("float16", 8, "shared.dyn", False),
)

MMA_store_16x16_f16_global_INTRIN = "mma_store_16x16_f16_global_"
TensorIntrin.register(
    MMA_store_16x16_f16_global_INTRIN, *get_mma_store_intrin("float16", 8, "global", True)
)

MMA_store_16x16_i32_global_INTRIN = "mma_store_16x16_i32_global_"
TensorIntrin.register(
    MMA_store_16x16_i32_global_INTRIN, *get_mma_store_intrin("int32", 8, "global", True)
)


def get_mma_intrin_group(
    load_scope: Literal["shared", "shared.dyn"],
    store_scope: Literal["global", "shared", "shared.dyn"],
    in_dtype: Literal["float16", "int8", "e4m3_float8", "e5m2_float8"],
    out_dtype: Literal["float16", "float32", "int32"],
    trans_a: bool,
    trans_b: bool,
    not_use_mma_store_intrinic: bool = True,
    store_to_smem_dtype: Optional[Literal["float16", "float32", "int32"]] = None,
) -> Dict[str, str]:
    """Get a group of intrinsics for mma tensor core with the given configurations

    Parameters
    ----------
    load_scope : Literal["shared", "shared.dyn"]
        The memory scope of the input buffer.

    store_scope : Literal["global", "shared", "shared.dyn"]
        The memory scope of the result buffer.

    in_dtype : str
        The input data type.

    out_dtype : str
        The output data dtype.

    trans_a : bool
        Whether the input matrix A is transposed.

    trans_b : bool
        Whether the input matrix B is transposed.

    not_use_mma_store_intrinic : bool
        Whether to not use the mma_store intrinsic. If True, use BufferStore stmts to store the
        result of mma. Otherwise, use mma_store intrinsic.

        This is because if we use mma_store intrinsic, during swizzling shared memory visits, our
        rearrangement scheme will involve areas accessed by different mma_store calls. This makes
        swizzling quite complex. But BufferStore will not face this problem.

    store_to_smem_dtype : Optional[Literal["float16", "float32", "int32"]]
        The dtype that we use to store from register to shared memory. By default it is out_dtype.

    Returns
    -------
    ret : Dict[str, str]
        A group of tensor intrinsics.
    """
    assert load_scope in ["shared", "shared.dyn"]
    assert store_scope in ["global", "shared", "shared.dyn"]
    assert in_dtype in ["float16", "int8", "e4m3_float8", "e5m2_float8"]
    assert out_dtype in ["float16", "float32", "int32"]

    shape = "16x16"

    dtype_mapping = {
        "float16": "f16",
        "float32": "f32",
        "int8": "i8",
        "e4m3_float8": "e4m3",
        "e5m2_float8": "e5m2",
        "int32": "i32",
    }
    a_dtype = dtype_mapping[in_dtype]
    b_dtype = dtype_mapping[in_dtype]
    out_dtype = dtype_mapping[out_dtype]

    # e.g. mma_fill_16x16_f32
    init_intrin = f"mma_fill_{shape}_{out_dtype}"

    # e.g. mma_ldmatrix_f16_a_trans_dyn, mma_ldmatrix_f16_b_trans_dyn
    trans_a = "_trans" if trans_a else ""
    trans_b = "_trans" if trans_b else ""
    load_scope = "_dyn" if load_scope == "shared.dyn" else ""
    load_a_intrin = f"mma_ldmatrix_{a_dtype}_a{trans_a}{load_scope}"
    load_b_intrin = f"mma_ldmatrix_{b_dtype}_b{trans_b}{load_scope}"

    # e.g. mma_f16f16f32_trans_a_trans_b
    trans_a_str = trans_a + "_a" if trans_a != "" else ""
    trans_b_str = trans_b + "_b" if trans_b != "" else ""
    compute_intrin = f"mma_{a_dtype}{b_dtype}{out_dtype}{trans_a_str}{trans_b_str}"

    # e.g. mma_store_16x16_f32_shared_dyn_simple_
    store_scope = store_scope.replace(".", "_")
    store_to_smem_dtype = dtype_mapping[store_to_smem_dtype] if store_to_smem_dtype else out_dtype
    suffix = "simple_" if not_use_mma_store_intrinic else ""
    store_intrin = f"mma_store_{shape}_{store_to_smem_dtype}_{store_scope}_{suffix}"

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
    }


######## WMMA intrinsics ########


def get_wmma_fragment_index(buffer, stride, m_dim, n_dim):
    """Compute wmma fragment index using elem_offset of the buffer"""
    frag_index_m = buffer.elem_offset // stride // m_dim
    frag_index_n = buffer.elem_offset % stride // n_dim

    num_fragments_per_row = stride // n_dim
    return frag_index_m * num_fragments_per_row + frag_index_n


def get_wmma_load_intrin(
    m_dim: int,
    n_dim: int,
    k_dim: int,
    dtype: str,
    shared_scope: str,
    is_b: bool,
    is_col_major: bool,
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_load intrins"""
    wmma_fragment_scope = f"wmma.matrix_{'b' if is_b else 'a'}"
    layout = "col_major" if is_col_major else "row_major"

    frag_m, frag_n = (k_dim, n_dim) if is_b else (m_dim, k_dim)
    if is_col_major:
        frag_m, frag_n = frag_n, frag_m
    offset_factor = frag_n

    @T.prim_func
    def wmma_load_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (frag_m, frag_n), dtype, align=64, offset_factor=offset_factor, scope=shared_scope
        )
        C = T.match_buffer(
            c,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=wmma_fragment_scope,
        )
        with T.block("root"):
            T.reads(A[0:frag_m, 0:frag_n])
            T.writes(C[0:frag_m, 0:frag_n])
            for i, j in T.grid(frag_m, frag_n):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=shared_scope,
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=wmma_fragment_scope,
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads(A[0:frag_m, 0:frag_n])
            T.writes(C[0:frag_m, 0:frag_n])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(C, d1, frag_m, frag_n),
                    A.access_ptr("r"),
                    s1,
                    layout,
                    dtype="handle",
                )
            )

    return wmma_load_desc, wmma_load_impl


def get_wmma_fill_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_fill intrins"""
    zero = IntImm("int32", 0).astype(dtype)
    offset_factor = n_dim

    @T.prim_func
    def wmma_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = zero

    @T.prim_func
    def wmma_fill_impl(c: T.handle) -> None:
        d1 = T.int32()
        d0 = T.int32()
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_fill_fragment(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(C, d1, m_dim, n_dim),
                    T.float32(0),
                    dtype="handle",
                )
            )

    return wmma_fill_desc, wmma_fill_impl


def get_wmma_store_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str, scope: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_store intrins"""
    offset_factor = n_dim

    @T.prim_func
    def wmma_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=offset_factor, scope=scope
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_store_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=scope,
            strides=[s1, s0],
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_store_matrix_sync(
                    A.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(A, d1, m_dim, n_dim),
                    C.access_ptr("w"),
                    s1,
                    "row_major",
                    dtype="handle",
                )
            )

    return wmma_store_desc, wmma_store_impl


def get_wmma_sync_intrin(
    m_dim: int, n_dim: int, k_dim: int, in_dtype: str, out_dtype: str, b_transposed: bool
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_sync intrins"""

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    b_shape_0, b_shape_1 = maybe_swap(k_dim, n_dim)

    A_offset_factor = k_dim
    B_offset_factor = b_shape_1
    out_offset_factor = n_dim

    @T.prim_func
    def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="wmma.matrix_a",
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="wmma.matrix_b",
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    B_index_0, B_index_1 = T.meta_var(maybe_swap(vkk, vjj))
                    C[vii, vjj] = C[vii, vjj] + maybe_cast(A[vii, vkk]) * maybe_cast(
                        B[B_index_0, B_index_1]
                    )

    @T.prim_func
    def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        a1 = T.int32()
        a0 = T.int32()
        b1 = T.int32()
        b0 = T.int32()
        c1 = T.int32()
        c0 = T.int32()

        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="wmma.matrix_a",
            strides=[a1, a0],
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="wmma.matrix_b",
            strides=[b1, b0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="wmma.accumulator",
            strides=[c1, c0],
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_mma_sync(
                    C.data,
                    get_wmma_fragment_index(C, c1, m_dim, n_dim),
                    A.data,
                    get_wmma_fragment_index(A, a1, m_dim, k_dim),
                    B.data,
                    get_wmma_fragment_index(B, b1, b_shape_0, b_shape_1),
                    C.data,
                    get_wmma_fragment_index(C, c1, m_dim, n_dim),
                    dtype="handle",
                )
            )

    return wmma_sync_desc, wmma_sync_impl


WMMA_SYNC_16x16x16_f16f16f32_INTRIN = "wmma_sync_16x16x16_f16f16f32"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float32", False),
)

WMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN = "wmma_sync_16x16x16_f16f16f32_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float32", True),
)

WMMA_SYNC_16x16x16_f16f16f16_INTRIN = "wmma_sync_16x16x16_f16f16f16"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float16", False),
)

WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN = "wmma_sync_16x16x16_f16f16f16_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float16", True),
)

WMMA_SYNC_16x16x16_s8s8s32_INTRIN = "wmma_sync_16x16x16_s8s8s32"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_s8s8s32_INTRIN, *get_wmma_sync_intrin(16, 16, 16, "int8", "int32", False)
)

WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN = "wmma_sync_16x16x16_s8s8s32_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "int8", "int32", True),
)

WMMA_SYNC_8x8x32_s4s4s32_TRANS_INTRIN = "wmma_sync_8x8x32_s4s4s32_trans"
TensorIntrin.register(
    WMMA_SYNC_8x8x32_s4s4s32_TRANS_INTRIN, *get_wmma_sync_intrin(8, 8, 32, "int4", "int32", True)
)

WMMA_LOAD_16x16x16_F16_A_INTRIN = "wmma_load_16x16x16_f16_a_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", False, False),
)

WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN = "wmma_load_16x16x16_f16_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", False, False),
)

WMMA_LOAD_16x16x16_F16_B_INTRIN = "wmma_load_16x16x16_f16_b_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", True, False),
)

WMMA_LOAD_16x16x16_F16_B_DYN_INTRIN = "wmma_load_16x16x16_f16_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", True, False),
)

WMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN = "wmma_load_16x16x16_f16_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", False, True),
)

WMMA_LOAD_16x16x16_F16_A_TRANS_DYN_INTRIN = "wmma_load_16x16x16_f16_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", False, True),
)

WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN = "wmma_load_16x16x16_f16_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", True, True),
)

WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN = "wmma_load_16x16x16_f16_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", True, True),
)

WMMA_LOAD_16x16x16_S8_A_INTRIN = "wmma_load_16x16x16_s8_a_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", False, False),
)

WMMA_LOAD_16x16x16_S8_A_DYN_INTRIN = "wmma_load_16x16x16_s8_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", False, False),
)

WMMA_LOAD_16x16x16_S8_B_INTRIN = "wmma_load_16x16x16_s8_b_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_INTRIN, *get_wmma_load_intrin(16, 16, 16, "int8", "shared", True, False)
)

WMMA_LOAD_16x16x16_S8_B_DYN_INTRIN = "wmma_load_16x16x16_s8_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", True, False),
)

WMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN = "wmma_load_16x16x16_s8_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", False, True),
)

WMMA_LOAD_16x16x16_S8_A_TRANS_DYN_INTRIN = "wmma_load_16x16x16_s8_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", False, True),
)

WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN = "wmma_load_16x16x16_s8_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", True, True),
)

WMMA_LOAD_16x16x16_S8_B_TRANS_DYN_INTRIN = "wmma_load_16x16x16_s8_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", True, True),
)

WMMA_LOAD_8x8x32_S4_A_INTRIN = "wmma_load_8x8x32_s4_a_shared"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_A_INTRIN, *get_wmma_load_intrin(8, 8, 32, "int4", "shared", False, False)
)

WMMA_LOAD_8x8x32_S4_A_DYN_INTRIN = "wmma_load_8x8x32_s4_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_A_DYN_INTRIN,
    *get_wmma_load_intrin(8, 8, 32, "int4", "shared.dyn", False, False),
)

WMMA_LOAD_8x8x32_S4_B_TRANS_INTRIN = "wmma_load_8x8x32_s4_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_B_TRANS_INTRIN,
    *get_wmma_load_intrin(8, 8, 32, "int4", "shared", True, True),
)

WMMA_LOAD_8x8x32_S4_B_TRANS_DYN_INTRIN = "wmma_load_8x8x32_s4_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(8, 8, 32, "int4", "shared.dyn", True, True),
)

WMMA_FILL_16x16x16_F32_INTRIN = "wmma_fill_16x16x16_f32"
TensorIntrin.register(WMMA_FILL_16x16x16_F32_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "float32"))

WMMA_FILL_16x16x16_F16_INTRIN = "wmma_fill_16x16x16_f16"
TensorIntrin.register(WMMA_FILL_16x16x16_F16_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "float16"))

WMMA_FILL_16x16x16_S32_INTRIN = "wmma_fill_16x16x16_s32"
TensorIntrin.register(WMMA_FILL_16x16x16_S32_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "int32"))

WMMA_FILL_8x8x32_S32_INTRIN = "wmma_fill_8x8x32_s32"
TensorIntrin.register(WMMA_FILL_8x8x32_S32_INTRIN, *get_wmma_fill_intrin(8, 8, 32, "int32"))

WMMA_STORE_16x16x16_F32_SHARED_INTRIN = "wmma_store_16x16x16_f32_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F32_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float32", "shared")
)

WMMA_STORE_16x16x16_F32_SHARED_DYN_INTRIN = "wmma_store_16x16x16_f32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(16, 16, 16, "float32", "shared.dyn"),
)

WMMA_STORE_16x16x16_F16_SHARED_INTRIN = "wmma_store_16x16x16_f16_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F16_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float16", "shared")
)

WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN = "wmma_store_16x16x16_f16_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(16, 16, 16, "float16", "shared.dyn"),
)

WMMA_STORE_16x16x16_S32_SHARED_INTRIN = "wmma_store_16x16x16_s32_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_S32_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "int32", "shared")
)

WMMA_STORE_16x16x16_S32_SHARED_DYN_INTRIN = "wmma_store_16x16x16_s32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_16x16x16_S32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(16, 16, 16, "int32", "shared.dyn"),
)

WMMA_STORE_8x8x32_S32_SHARED_INTRIN = "wmma_store_8x8x32_s32_shared"
TensorIntrin.register(
    WMMA_STORE_8x8x32_S32_SHARED_INTRIN, *get_wmma_store_intrin(8, 8, 32, "int32", "shared")
)

WMMA_STORE_8x8x32_S32_SHARED_DYN_INTRIN = "wmma_store_8x8x32_s32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_8x8x32_S32_SHARED_DYN_INTRIN, *get_wmma_store_intrin(8, 8, 32, "int32", "shared.dyn")
)

WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN = "wmma_store_16x16x16_f32_global"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float32", "global")
)

WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN = "wmma_store_16x16x16_f16_global"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float16", "global")
)

WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN = "wmma_store_16x16x16_s32_global"
TensorIntrin.register(
    WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN, *get_wmma_store_intrin(16, 16, 16, "int32", "global")
)

WMMA_STORE_8x8x32_S32_GLOBAL_INTRIN = "wmma_store_8x8x32_s32_global"
TensorIntrin.register(
    WMMA_STORE_8x8x32_S32_GLOBAL_INTRIN, *get_wmma_store_intrin(8, 8, 32, "int32", "global")
)


def get_wmma_intrin_group(
    load_scope: Literal["shared", "shared.dyn"],
    store_scope: Literal["global", "shared", "shared.dyn"],
    in_dtype: str,
    out_dtype: str,
    trans_b: bool,
) -> Dict[str, str]:
    """Get a group of intrinsics for wmma tensor core with the given configurations

    Parameters
    ----------
    load_scope : Literal["shared", "shared.dyn"]
        The memory scope of the input buffer.

    store_scope : Literal["global", "shared", "shared.dyn"]
        The memory scope of the result buffer.

    in_dtype : str
        The input data type.

    out_dtype : str
        The output data dtype.

    trans_b : bool
        Whether the input matrix B is transposed.

    Returns
    -------
    ret : Dict[str, str]
        A group of tensor intrinsics.
    """
    assert load_scope in ["shared", "shared.dyn"]
    assert store_scope in ["global", "shared", "shared.dyn"]
    assert in_dtype in ["float16", "int8"]
    assert out_dtype in ["float16", "float32", "int32"]

    shape = "16x16x16"
    in_dtype = "f16" if in_dtype == "float16" else "s8"
    out_dtype = "f16" if out_dtype == "float16" else "f32" if out_dtype == "float32" else "s32"
    # convert "shared.dyn" to "shared_dyn"
    load_scope = load_scope.replace(".", "_")
    store_scope = store_scope.replace(".", "_")
    trans_a = ""
    trans_b = "_trans" if trans_b else ""

    # e.g. wmma_load_16x16x16_f16_a_shared
    load_a_intrin = f"wmma_load_{shape}_{in_dtype}_a{trans_a}_{load_scope}"
    # e.g. wmma_load_16x16x16_f16_b_trans_shared_dyn
    load_b_intrin = f"wmma_load_{shape}_{in_dtype}_b{trans_b}_{load_scope}"
    # e.g. wmma_sync_16x16x16_f16f16f32_trans
    compute_intrin = f"wmma_sync_{shape}_{in_dtype}{in_dtype}{out_dtype}{trans_b}"
    # e.g. wmma_fill_16x16x16_f16
    init_intrin = f"wmma_fill_{shape}_{out_dtype}"
    # e.g. wmma_store_16x16x16_f16_shared_dyn
    store_intrin = f"wmma_store_{shape}_{out_dtype}_{store_scope}"

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
    }


######## MMA intrinsics ########


def get_index_A(elem_offset, stride):
    i = elem_offset // stride
    j = elem_offset % stride
    stride_b = stride // 8
    bi = i // 32
    bj = j // 8
    no = bi * stride_b + bj
    return no * 8 + (i % 32) // 16 * 4


def get_index_B(elem_offset, stride):
    i = elem_offset // stride
    j = elem_offset % stride
    stride_b = stride // 32
    bi = i // 8
    bj = j // 32
    no = bi * stride_b + bj
    return no * 8 + (j % 32) // 8 * 2


def get_index_C(elem_offset, stride):
    i = elem_offset // stride
    j = elem_offset % stride
    stride_b = stride // 8
    bi = i // 8
    bj = j // 8
    return (bi // 2) * 2 * stride_b + bi % 2 + bj * 2


def get_mma_init_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of mma init intrins"""
    del k_dim  # unused
    zero = IntImm("int32", 0).astype(dtype)
    assert m_dim % 8 == 0 and n_dim % 4 == 0, "m_dim and n_dim must be multiple of 8 and 4"
    assert dtype in ["float16", "float32"]
    assert n_dim // 4 * int(dtype[-2:]) <= 128, "n_dim vectorize failed"

    @T.prim_func
    def mma_init_desc(c: T.handle) -> None:
        dst = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=1, scope="m16n8k8.matrixC"
        )
        with T.block("root"):
            T.reads()
            T.writes(dst[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("init"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    dst[vi, vj] = zero

    @T.prim_func
    def mma_init_impl(c: T.handle) -> None:
        dst = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=1, scope="m16n8k8.matrixC"
        )

        with T.block("root"):
            T.reads()
            T.writes(dst[0:m_dim, 0:n_dim])

            for tx in T.thread_binding(0, WARP_SIZE, "threadIdx.x"):
                for b in range(m_dim // 8):
                    for v in T.vectorized(n_dim // 4):
                        dst[b * 8 + tx // 4, (tx % 4) * (n_dim // 4) + v] = zero

    return mma_init_desc, mma_init_impl


def get_mma_load_intrin(
    m_dim: int,
    n_dim: int,
    k_dim: int,
    dtype: str,
    shared_scope: str,
    is_b: bool,
    is_col_major: bool,
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of mma ldmatrix intrins"""
    mma_fragment_scope = f"m16n8k8.matrix{'B' if is_b else 'A'}"
    frag_m, frag_n = (k_dim, n_dim) if is_b else (m_dim, k_dim)
    trans = (not is_col_major) if is_b else is_col_major
    if is_col_major:
        frag_m, frag_n = frag_n, frag_m
    get_index = get_index_B if is_b else get_index_A
    get_tx_index = (
        (lambda tx, s0: (tx % 8) * s0 + (tx // 8) * 8) if trans else (lambda tx, s0: tx * s0)
    )

    @T.prim_func
    def mma_load_desc(a: T.handle, c: T.handle) -> None:
        src = T.match_buffer(
            a, (frag_m, frag_n), dtype, align=64, offset_factor=1, scope=shared_scope
        )
        dst = T.match_buffer(
            c, (frag_m, frag_n), dtype, align=64, offset_factor=1, scope=mma_fragment_scope
        )

        with T.block("root"):
            T.reads(src[0:frag_m, 0:frag_n])
            T.writes(dst[0:frag_m, 0:frag_n])
            for i, j in T.grid(frag_m, frag_n):
                with T.block("root"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    dst[vi, vj] = src[vi, vj]

    @T.prim_func
    def mma_load_impl(a: T.handle, c: T.handle) -> None:
        s0 = T.int32()
        s1 = T.int32()
        src = T.match_buffer(
            a,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=1,
            scope=shared_scope,
            strides=[s0, s1],
        )
        d0 = T.int32()
        d1 = T.int32()
        dst = T.match_buffer(
            c,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=1,
            scope=mma_fragment_scope,
            strides=[d0, d1],
        )

        with T.block("root"):
            T.reads(src[0:frag_m, 0:frag_n])
            T.writes(dst[0:frag_m, 0:frag_n])

            for tx in T.thread_binding(0, WARP_SIZE, "threadIdx.x"):
                T.evaluate(
                    T.ptx_ldmatrix(
                        trans,
                        4,  # Always load 4 matrices
                        ".b16",
                        dst.data,
                        get_index(dst.elem_offset, d0),
                        src.access_ptr("r"),
                        get_tx_index(tx, s0),
                        dtype=dtype,
                    )
                )

    return mma_load_desc, mma_load_impl


def get_mma_sync_intrin(
    m_dim: int, n_dim: int, k_dim: int, in_dtype: str, out_dtype: str, b_transposed: bool
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of mma sync intrins"""

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    B_shape_0, B_shape_1 = maybe_swap(k_dim, n_dim)

    @T.prim_func
    def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (m_dim, k_dim), in_dtype, align=64, offset_factor=1, scope="m16n8k8.matrixA"
        )
        B = T.match_buffer(
            b, (B_shape_0, B_shape_1), in_dtype, align=64, offset_factor=1, scope="m16n8k8.matrixB"
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), out_dtype, align=64, offset_factor=1, scope="m16n8k8.matrixC"
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:B_shape_0, 0:B_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block("m16n8k8_sync"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    B_index_0, B_index_1 = T.meta_var(maybe_swap(vk, vj))
                    C[vi, vj] = C[vi, vj] + maybe_cast(A[vi, vk]) * maybe_cast(
                        B[B_index_0, B_index_1]
                    )

    @T.prim_func
    def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        a0 = T.int32()
        a1 = T.int32()
        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=1,
            scope="m16n8k8.matrixA",
            strides=[a0, a1],
        )
        b0 = T.int32()
        b1 = T.int32()
        B = T.match_buffer(
            b,
            (B_shape_0, B_shape_1),
            in_dtype,
            align=64,
            offset_factor=1,
            scope="m16n8k8.matrixB",
            strides=[b0, b1],
        )
        c0 = T.int32()
        c1 = T.int32()
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=1,
            scope="m16n8k8.matrixC",
            strides=[c0, c1],
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:B_shape_0, 0:B_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.ptx_mma(
                    f"m{m_dim}n{n_dim}k{k_dim}",
                    "row",
                    "col",
                    in_dtype,
                    in_dtype,
                    out_dtype,
                    A.data,
                    get_index_A(A.elem_offset, a0),
                    B.data,
                    get_index_B(B.elem_offset, b0),
                    C.data,
                    get_index_C(C.elem_offset, c0),
                    False,
                    dtype=out_dtype,
                )
            )

    return mma_sync_desc, mma_sync_impl


def get_mma_store_dummy_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Disable mma store intrin for now."""
    del k_dim  # unused

    @T.prim_func
    def mma_store_desc(a: T.handle, c: T.handle) -> None:
        src = T.match_buffer(
            a, (m_dim, n_dim), dtype, align=64, offset_factor=1, scope="m16n8k8.matrixC"
        )
        dst = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=1, scope="shared.dyn"
        )

        with T.block("root"):
            T.reads(src[0:m_dim, 0:n_dim])
            T.writes(dst[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("m16n8k8_store"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    dst[vi, vj] = src[vi, vj]

    return mma_store_desc, mma_store_desc


TensorIntrin.register("mma_init_m16n8k8_f16", *get_mma_init_intrin(16, 8, 8, "float16"))
TensorIntrin.register("mma_init_m16n8k8_f32", *get_mma_init_intrin(16, 8, 8, "float32"))

TensorIntrin.register(
    "mma_load_m16n8k8_f16_A_shared_dyn",
    *get_mma_load_intrin(32, 32, 8, "float16", "shared.dyn", False, False),
)
TensorIntrin.register(
    "mma_load_m16n8k8_f16_B_shared_dyn",
    *get_mma_load_intrin(32, 32, 8, "float16", "shared.dyn", True, False),
)

TensorIntrin.register(
    "mma_sync_m16n8k8_f16f16f16", *get_mma_sync_intrin(16, 8, 8, "float16", "float16", False)
)
TensorIntrin.register(
    "mma_sync_m16n8k8_f16f16f32", *get_mma_sync_intrin(16, 8, 8, "float16", "float32", False)
)

TensorIntrin.register(
    "mma_store_m16n8k8_f16_global", *get_mma_store_dummy_intrin(16, 8, 8, "float16")
)
TensorIntrin.register(
    "mma_store_m16n8k8_f32_global", *get_mma_store_dummy_intrin(16, 8, 8, "float32")
)


@register_func("tir.index_map_m16n8k8.matrixC")
def index_map_m16n8k8_matrixC(ind):
    i, j = ind[0], ind[1]
    return convert([(i // 8) // 2, j // 8, (i // 8) % 2, (j % 8) % 2])
