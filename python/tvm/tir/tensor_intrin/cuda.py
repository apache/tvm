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
from typing import Tuple, Dict
from tvm.script import tir as T
from tvm.tir.function import PrimFunc
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


def get_ldmatrix_intrin(k_dim, dtype, is_b, transposed, shared_scope="shared"):
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
            shared_handle,
            shmem_shape,
            dtype,
            align=64,
            offset_factor=16,
            scope=shared_scope,
        )
        warp = T.match_buffer(
            warp_handle, (WARP_SIZE, local_size), dtype, align=64, offset_factor=16, scope="warp"
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
            align=64,
            offset_factor=16,
            scope=shared_scope,
            strides=[s0, s1],
        )
        warp = T.match_buffer(
            warp_handle, (WARP_SIZE, local_size), dtype, align=64, offset_factor=16, scope="warp"
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
            a, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, align=64, offset_factor=16, scope="warp"
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
            a, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, align=64, offset_factor=16, scope="warp"
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

LDMATRIX_16x16_A_DYN_INTRIN = "mma.ldmatrix_16x16_a_dyn"
TensorIntrin.register(
    LDMATRIX_16x16_A_DYN_INTRIN, *get_ldmatrix_intrin(16, "float16", False, False, "shared.dyn")
)

LDMATRIX_16x16_B_DYN_INTRIN = "mma.ldmatrix_16x16_b_dyn"
TensorIntrin.register(
    LDMATRIX_16x16_B_DYN_INTRIN, *get_ldmatrix_intrin(16, "float16", True, False, "shared.dyn")
)

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


######## WMMA intrinsics ########


def get_wmma_fragment_index(buffer, m_dim, n_dim):
    """Compute wmma fragment index using elem_offset of the buffer"""
    frag_size = lift(m_dim * n_dim)
    return buffer.elem_offset // frag_size + (buffer.elem_offset % frag_size) // n_dim


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
    wmma_fragment_scope = "wmma.matrix_{}".format("b" if is_b else "a")
    layout = "col_major" if is_col_major else "row_major"

    @T.prim_func
    def wmma_load_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=shared_scope)
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=wmma_fragment_scope
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.var("int32")
        s0 = T.var("int32")
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=16,
            scope=shared_scope,
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=wmma_fragment_scope
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(C, m_dim, n_dim),
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

    @T.prim_func
    def wmma_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope="wmma.accumulator"
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
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope="wmma.accumulator"
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
                    get_wmma_fragment_index(C, m_dim, n_dim),
                    T.float32(0),
                    dtype="handle",
                )
            )

    return wmma_fill_desc, wmma_fill_impl


def get_wmma_store_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str, scope: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_store intrins"""

    @T.prim_func
    def wmma_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope="wmma.accumulator"
        )
        C = T.match_buffer(c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=scope)
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_store_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.var("int32")
        s0 = T.var("int32")
        A = T.match_buffer(
            a, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope="wmma.accumulator"
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=scope, strides=[s1, s0]
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
                    get_wmma_fragment_index(A, m_dim, n_dim),
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

    @T.prim_func
    def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (m_dim, k_dim), in_dtype, align=64, offset_factor=16, scope="wmma.matrix_a"
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=16,
            scope="wmma.matrix_b",
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), out_dtype, align=64, offset_factor=16, scope="wmma.accumulator"
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    B_index_0, B_index_1 = maybe_swap(vkk, vjj)
                    C[vii, vjj] = C[vii, vjj] + maybe_cast(A[vii, vkk]) * maybe_cast(
                        B[B_index_0, B_index_1]
                    )

    @T.prim_func
    def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (m_dim, k_dim), in_dtype, align=64, offset_factor=16, scope="wmma.matrix_a"
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=16,
            scope="wmma.matrix_b",
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), out_dtype, align=64, offset_factor=16, scope="wmma.accumulator"
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_mma_sync(
                    C.data,
                    get_wmma_fragment_index(C, m_dim, n_dim),
                    A.data,
                    get_wmma_fragment_index(A, m_dim, k_dim),
                    B.data,
                    get_wmma_fragment_index(B, b_shape_0, b_shape_1),
                    C.data,
                    get_wmma_fragment_index(C, m_dim, n_dim),
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
    WMMA_SYNC_16x16x16_s8s8s32_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "int8", "int32", False),
)

WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN = "wmma_sync_16x16x16_s8s8s32_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "int8", "int32", True),
)

WMMA_LOAD_16x16x16_F16_A_INTRIN = "wmma_load_16x16x16_f16_a"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", False, False),
)

WMMA_LOAD_16x16x16_F16_B_INTRIN = "wmma_load_16x16x16_f16_b"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", True, False),
)

WMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN = "wmma_load_16x16x16_f16_a_trans"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", False, True),
)

WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN = "wmma_load_16x16x16_f16_b_trans"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", True, True),
)

WMMA_LOAD_16x16x16_S8_A_INTRIN = "wmma_load_16x16x16_s8_a"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", False, False),
)

WMMA_LOAD_16x16x16_S8_B_INTRIN = "wmma_load_16x16x16_s8_b"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", True, False),
)

WMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN = "wmma_load_16x16x16_s8_a_trans"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", False, True),
)

WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN = "wmma_load_16x16x16_s8_b_trans"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", True, True),
)


WMMA_FILL_16x16x16_F32_INTRIN = "wmma_fill_16x16x16_f32"
TensorIntrin.register(WMMA_FILL_16x16x16_F32_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "float32"))

WMMA_FILL_16x16x16_F16_INTRIN = "wmma_fill_16x16x16_f16"
TensorIntrin.register(WMMA_FILL_16x16x16_F16_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "float16"))

WMMA_FILL_16x16x16_S32_INTRIN = "wmma_fill_16x16x16_s32"
TensorIntrin.register(WMMA_FILL_16x16x16_S32_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "int32"))

WMMA_STORE_16x16x16_F32_SHARED_INTRIN = "wmma_store_16x16x16_f32_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F32_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float32", "shared")
)

WMMA_STORE_16x16x16_F16_SHARED_INTRIN = "wmma_store_16x16x16_f16_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F16_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float16", "shared")
)

WMMA_STORE_16x16x16_S32_SHARED_INTRIN = "wmma_store_16x16x16_s32_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_S32_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "int32", "shared")
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


def get_wmma_intrin_group(
    store_scope: str, in_dtype: str, out_dtype: str, trans_b: bool
) -> Dict[str, str]:
    """Get a group of intrinsics for wmma tensor core with the given configurations

    Parameters
    ----------
    store_scope : str
        Must be one of ["global", "shared"]. The memory scope of the result buffer.

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
    assert store_scope in ["global", "shared"]
    assert in_dtype in ["float16", "int8"]
    assert out_dtype in ["float16", "float32", "int32"]

    load_a_intrins = {
        "float16": WMMA_LOAD_16x16x16_F16_A_INTRIN,
        "int8": WMMA_LOAD_16x16x16_S8_A_INTRIN,
    }
    load_b_intrins = {
        "float16": WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN
        if trans_b
        else WMMA_LOAD_16x16x16_F16_B_INTRIN,
        "int8": WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN if trans_b else WMMA_LOAD_16x16x16_S8_B_INTRIN,
    }
    compute_intrins = {
        "float16": WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN
        if trans_b
        else WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
        "float32": WMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN
        if trans_b
        else WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
        "int32": WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN
        if trans_b
        else WMMA_SYNC_16x16x16_s8s8s32_INTRIN,
    }
    init_intrins = {
        "float16": WMMA_FILL_16x16x16_F16_INTRIN,
        "float32": WMMA_FILL_16x16x16_F32_INTRIN,
        "int32": WMMA_FILL_16x16x16_S32_INTRIN,
    }
    store_intrins = {
        "float16": WMMA_STORE_16x16x16_F16_SHARED_INTRIN
        if store_scope == "shared"
        else WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
        "float32": WMMA_STORE_16x16x16_F32_SHARED_INTRIN
        if store_scope == "shared"
        else WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
        "int32": WMMA_STORE_16x16x16_S32_SHARED_INTRIN
        if store_scope == "shared"
        else WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN,
    }
    return {
        "init": init_intrins[out_dtype],
        "load_a": load_a_intrins[in_dtype],
        "load_b": load_b_intrins[in_dtype],
        "compute": compute_intrins[out_dtype],
        "store": store_intrins[out_dtype],
    }
