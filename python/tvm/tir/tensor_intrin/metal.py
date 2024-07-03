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
"""Intrinsics for tensorization on Apple GPU."""
from typing import Dict, Literal, Tuple

from tvm.script import tir as T
from tvm.tir import Buffer, PrimExpr, PrimFunc, TensorIntrin

######## simdgroup matrix intrinsics ########


def get_simdgroup_index(buffer: Buffer, stride: PrimExpr, col: int, row: int):
    """Compute simdgroup index using elem_offset of the buffer"""

    # NOTE: Need further check the usage between `col`` and `row`
    # Currently, Metal only supports 8x8, which means the values of `col` and `row` are the same
    frag_index_m = buffer.elem_offset // stride // col
    frag_index_n = buffer.elem_offset % stride // row

    num_fragments_per_row = stride // row
    return frag_index_m * num_fragments_per_row + frag_index_n


def get_make_filled_simdgroup_matrix_intrin(
    dtype: str, col: int = 8, row: int = 8
) -> Tuple[PrimFunc, PrimFunc]:
    @T.prim_func
    def desc(a: T.handle) -> None:
        A = T.match_buffer(a, (col, row), dtype, scope="metal.simdgroup", offset_factor=1)
        with T.block("root"):
            T.reads()
            T.writes(A[0:col, 0:row])
            for i, j in T.grid(col, row):
                with T.block("init"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    A[vi, vj] = T.float32(0)

    @T.prim_func
    def impl(a: T.handle) -> None:
        d0, d1 = T.int32(), T.int32()
        A = T.match_buffer(
            a, (col, row), dtype, scope="metal.simdgroup", strides=[d1, d0], offset_factor=1
        )
        with T.block("root"):
            T.reads()
            T.writes(A[0:col, 0:row])
            T.make_filled_simdgroup_matrix(
                A.data,
                index=get_simdgroup_index(A, d1, col, row),
                value=T.float32(0),
                col=col,
                row=row,
            )

    return desc, impl


def get_simdgroup_load_intrin(
    dtype: str,
    scope: Literal["global", "shared"],
    col: int = 8,
    row: int = 8,
    transpose_matrix: bool = False,
) -> Tuple[PrimFunc, PrimFunc]:
    align = col * row

    @T.prim_func
    def desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (col, row), dtype, align=align, scope=scope, offset_factor=1)
        C = T.match_buffer(
            c, (col, row), dtype, align=align, scope="metal.simdgroup", offset_factor=1
        )
        with T.block("root"):
            T.reads(A[0:col, 0:row])
            T.writes(C[0:col, 0:row])
            for i, j in T.grid(col, row):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    if transpose_matrix:
                        # C[vii, vjj] = A[vjj, vii]
                        C[vjj, vii] = A[vii, vjj]
                    else:
                        C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def impl(a: T.handle, c: T.handle) -> None:
        s0, s1, d0, d1 = T.int32(), T.int32(), T.int32(), T.int32()
        A = T.match_buffer(
            a,
            (col, row),
            dtype,
            align=align,
            scope=scope,
            strides=[s1, s0],
            offset_factor=1,
        )
        C = T.match_buffer(
            c,
            (col, row),
            dtype,
            align=align,
            scope="metal.simdgroup",
            strides=[d1, d0],
            offset_factor=1,
        )
        with T.block("root"):
            T.reads(A[0:col, 0:row])
            T.writes(C[0:col, 0:row])
            T.simdgroup_load(
                C.data,
                index=get_simdgroup_index(C, d1, col, row),
                ptr=A.access_ptr("r"),
                stride=s1,
                col=col,
                row=row,
                transpose_matrix=transpose_matrix,
            )

    return desc, impl


def get_simdgroup_store_intrin(
    dtype: str,
    scope: Literal["global", "shared"],
    col: int = 8,
    row: int = 8,
    transpose_matrix: bool = False,
) -> Tuple[PrimFunc, PrimFunc]:
    align = col * row

    @T.prim_func
    def desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (col, row), dtype, align=align, scope="metal.simdgroup", offset_factor=1
        )
        C = T.match_buffer(c, (col, row), dtype, align=align, scope=scope, offset_factor=1)
        with T.block("root"):
            T.reads(A[0:col, 0:row])
            T.writes(C[0:col, 0:row])
            for i, j in T.grid(col, row):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    if transpose_matrix:
                        C[vjj, vii] = A[vii, vjj]
                    else:
                        C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def impl(a: T.handle, c: T.handle) -> None:
        s0, s1, d0, d1 = T.int32(), T.int32(), T.int32(), T.int32()
        A = T.match_buffer(
            a,
            (col, row),
            dtype,
            align=align,
            scope="metal.simdgroup",
            strides=[s1, s0],
            offset_factor=1,
        )
        C = T.match_buffer(
            c, (col, row), dtype, align=align, scope=scope, strides=[d1, d0], offset_factor=1
        )
        with T.block("root"):
            T.reads(A[0:col, 0:row])
            T.writes(C[0:col, 0:row])
            T.simdgroup_store(
                A.data,
                index=get_simdgroup_index(A, s1, col, row),
                ptr=C.access_ptr("w"),
                stride=d1,
                col=col,
                row=row,
                transpose_matrix=transpose_matrix,
            )

    return desc, impl


def get_simdgroup_multiply_accumulate_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str
) -> Tuple[PrimFunc, PrimFunc]:
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_dim, k_dim), dtype, scope="metal.simdgroup", offset_factor=1)
        B = T.match_buffer(b, (k_dim, n_dim), dtype, scope="metal.simdgroup", offset_factor=1)
        C = T.match_buffer(c, (m_dim, n_dim), dtype, scope="metal.simdgroup", offset_factor=1)
        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:k_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    C[vii, vjj] += A[vii, vkk] * B[vkk, vjj]

    @T.prim_func
    def impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        a0, a1, b0, b1, c0, c1 = T.int32(), T.int32(), T.int32(), T.int32(), T.int32(), T.int32()
        A = T.match_buffer(
            a, (m_dim, k_dim), dtype, scope="metal.simdgroup", strides=[a1, a0], offset_factor=1
        )
        B = T.match_buffer(
            b, (k_dim, n_dim), dtype, scope="metal.simdgroup", strides=[b1, b0], offset_factor=1
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, scope="metal.simdgroup", strides=[c1, c0], offset_factor=1
        )
        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:k_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            T.simdgroup_multiply_accumulate(
                C.data,
                get_simdgroup_index(C, c1, m_dim, n_dim),
                A.data,
                get_simdgroup_index(A, a1, m_dim, k_dim),
                B.data,
                get_simdgroup_index(B, b1, k_dim, n_dim),
                C.data,
                get_simdgroup_index(C, c1, m_dim, n_dim),
            )

    return desc, impl


# Make filled simdgroup matrix intrinsics

SIMDGROUP_MAKE_FILLED_8x8x8_f16_INTRIN = "simdgroup_make_filled_8x8x8_f16"
TensorIntrin.register(
    SIMDGROUP_MAKE_FILLED_8x8x8_f16_INTRIN,
    *get_make_filled_simdgroup_matrix_intrin("float16", 8, 8),
)

SIMDGROUP_FILLED_8x8x8_f32_INTRIN = "simdgroup_fill_8x8x8_f32"
TensorIntrin.register(
    SIMDGROUP_FILLED_8x8x8_f32_INTRIN, *get_make_filled_simdgroup_matrix_intrin("float32", 8, 8)
)

SIMDGROUP_FILLED_8x8x8_bf16_INTRIN = "simdgroup_fill_8x8x8_bf16"
TensorIntrin.register(
    SIMDGROUP_FILLED_8x8x8_bf16_INTRIN, *get_make_filled_simdgroup_matrix_intrin("bfloat16", 8, 8)
)

# Load intrinsics

SIMDGROUP_LOAD_8x8x8_f16_SHARED_INTRIN = "simdgroup_load_8x8x8_f16_shared"
TensorIntrin.register(
    SIMDGROUP_LOAD_8x8x8_f16_SHARED_INTRIN,
    *get_simdgroup_load_intrin("float16", "shared", 8, 8, False),
)

SIMDGROUP_LOAD_8x8x8_f16_SHARED_TRANS_INTRIN = "simdgroup_load_8x8x8_f16_shared_trans"
TensorIntrin.register(
    SIMDGROUP_LOAD_8x8x8_f16_SHARED_TRANS_INTRIN,
    *get_simdgroup_load_intrin("float16", "shared", 8, 8, True),
)

# Store intrinsics

SIMDGROUP_STORE_8x8x8_f16_GLOBAL_INTRIN = "simdgroup_store_8x8x8_f16_global"
TensorIntrin.register(
    SIMDGROUP_STORE_8x8x8_f16_GLOBAL_INTRIN,
    *get_simdgroup_store_intrin("float16", "global", 8, 8, False),
)

SIMDGROUP_STORE_8x8x8_f16_SHARED_INTRIN = "simdgroup_store_8x8x8_f16_shared"
TensorIntrin.register(
    SIMDGROUP_STORE_8x8x8_f16_SHARED_INTRIN,
    *get_simdgroup_store_intrin("float16", "shared", 8, 8, False),
)
# Multiply accumulate intrinsics

SIMDGROUP_MULTI_ACC_8x8x8_f16_INTRIN = "simdgroup_multiply_accumulate_8x8x8_f16"
TensorIntrin.register(
    SIMDGROUP_MULTI_ACC_8x8x8_f16_INTRIN,
    *get_simdgroup_multiply_accumulate_intrin(8, 8, 8, "float16"),
)


def get_simdgroup_intrin_group(
    load_scope: Literal["shared"],
    store_scope: Literal["global", "shared"],
    dtype: str,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Dict[str, str]:
    """Get a group of intrinsics for tensorization on Apple GPU.

    Parameters
    ----------
    load_scope : Literal["shared"]
        The memory scope of the input buffer.

    store_scope : Literal["global", "shared"]
        The memory scope of the result buffer.

    dtype : str
        The data type of the input and output buffers.

    trans_a : bool
        Whether the input matrix A is transposed.

    trans_b : bool
        Whether the input matrix B is transposed.

    Returns
    -------
    ret : Dict[str, str]
        A group of tensor intrinsics.
    """
    assert load_scope in ["shared"]
    assert store_scope in ["global", "shared"]
    assert dtype in ["float16", "bfloat16", "float32"]

    shape = "8x8x8"
    dtype = "f16" if dtype == "float16" else "bf16" if dtype == "bfloat16" else "f32"
    trans_a = "_trans" if trans_a else ""
    trans_b = "_trans" if trans_b else ""

    # e.g. simdgroup_load_8x8x8_f16_shared
    load_a_intrin = f"simdgroup_load_{shape}_{dtype}_{load_scope}{trans_a}"
    # e.g. simdgroup_load_8x8x8_f16_shared_trans
    load_b_intrin = f"simdgroup_load_{shape}_{dtype}_{load_scope}{trans_b}"
    # e.g. simdgroup_multiply_accumulate_8x8x8_f16
    compute_intrin = f"simdgroup_multiply_accumulate_{shape}_{dtype}"
    # e.g. simdgroup_make_filled_8x8x8_f16
    init_intrin = f"simdgroup_make_filled_{shape}_{dtype}"
    # e.g. simdgroup_store_8x8x8_f16_global
    store_intrin = f"simdgroup_store_{shape}_{dtype}_{store_scope}"

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
    }
