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
"""Intrinsics for RDNA (gfx11) WMMA tensorization, wave32.

`v_wmma_f32_16x16x16_f16` computes D[16x16] = A[16x16] x B[16x16] + C on one wave32 wavefront.
Fragment layout per lane (verified on gfx1151):
  A: lane l holds row l % 16, all 16 k values (lanes 16-31 repeat lanes 0-15)
  B: lane l holds column l % 16, all 16 k values (lanes 16-31 repeat lanes 0-15)
  C/D: lane l holds 8 values, rows 2 * i + l // 16 (i = 0..7) of column l % 16
As in rocm.py (MFMA), descriptors use the logical (lane, value) layout of the whole wavefront and
implementations bind (1, values_per_lane) local storage per lane. The A/B descriptors describe the
16 distinct fragment lanes; the implementations run on all 32 lanes.
"""

from tvm.s_tir import script as Ts
from tvm_tirx.ir import IntImm
from tvm_tirx.script import tirx as T

from .. import TensorIntrin

WARP_SIZE = 32
M_DIM = N_DIM = K_DIM = 16
AB_LOCAL = 16  # values per lane for A and B fragments
AB_LANES = 16  # distinct A/B fragment lanes; lanes 16-31 hold a copy (the descriptors describe 16)
C_LOCAL = 8  # values per lane for the accumulator fragment


def shared_16x16_to_local_32x16_layout_A(i, k):
    return i, k


def thread_id_shared_access_32x16_to_16x16_layout_A(thread_id, local_id):
    return thread_id % 16, local_id


def shared_16x16_to_local_32x16_layout_B(k, j):
    return j, k


def thread_id_shared_access_32x16_to_16x16_layout_B(thread_id, local_id):
    return local_id, thread_id % 16


def shared_16x16_to_local_32x8_layout_C(i, j):
    return j + 16 * (i % 2), i // 2


def thread_id_shared_access_32x8_to_16x16_layout_C(thread_id, local_id):
    return 2 * local_id + thread_id // 16, thread_id % 16


def get_wmma_fill_intrin(dtype="float32"):
    zero = IntImm("int32", 0).astype(dtype)

    @Ts.function
    def fill_desc(C_warp: T.Tensor([WARP_SIZE, C_LOCAL], dtype=dtype, scope="local")) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(C_warp[0:WARP_SIZE, 0:C_LOCAL])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with Ts.sblock("C_warp"):
                    i, j = Ts.axis.remap("SS", [i0, i1])
                    w = shared_16x16_to_local_32x8_layout_C(i, j)
                    Ts.reads()
                    Ts.writes(C_warp[w[0], w[1]])
                    C_warp[w[0], w[1]] = zero

    fill_impl_offset = T.dynamic("wmma_fill_impl_offset", "int32")

    @Ts.function
    def fill_impl(
        C_warp: T.Tensor(
            [1, C_LOCAL],
            dtype=dtype,
            scope="local",
            layout=T.TileLayout(T.S[[1, C_LOCAL]] + (fill_impl_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(C_warp[0:1, 0:C_LOCAL])
            _tx = T.launch_thread("threadIdx.x", WARP_SIZE)
            for local_id in T.serial(0, C_LOCAL):
                C_warp[0, local_id] = zero

    return fill_desc, fill_impl


def get_wmma_load_intrin(dtype="float16", scope="shared", is_b=False, transposed=False):
    """A: memory [M, K]. B: memory [K, N], or [N, K] when `transposed` (weights stored N x K)."""
    if not is_b:
        memory_shape, index_map, reverse = (
            (M_DIM, K_DIM),
            shared_16x16_to_local_32x16_layout_A,
            thread_id_shared_access_32x16_to_16x16_layout_A,
        )
    elif transposed:  # B^T stored [N, K]: lane l reads row l % 16, like A
        memory_shape, index_map, reverse = (
            (N_DIM, K_DIM),
            shared_16x16_to_local_32x16_layout_A,
            thread_id_shared_access_32x16_to_16x16_layout_A,
        )
    else:
        memory_shape, index_map, reverse = (
            (K_DIM, N_DIM),
            shared_16x16_to_local_32x16_layout_B,
            thread_id_shared_access_32x16_to_16x16_layout_B,
        )
    row_dim, col_dim = memory_shape
    desc_reg_off = T.dynamic("wmma_load_desc_reg_offset", "int32")
    desc_mem_off = T.dynamic("wmma_load_desc_memory_offset", "int32")

    @Ts.function
    def load_desc(
        reg: T.Tensor(
            (AB_LANES, AB_LOCAL),
            dtype,
            scope="local",
            layout=T.TileLayout(T.S[(AB_LANES, AB_LOCAL)] + (desc_reg_off)),
        ),
        memory: T.Tensor(
            memory_shape,
            dtype,
            scope=scope,
            layout=T.TileLayout(T.S[memory_shape] + (desc_mem_off)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(memory[0:row_dim, 0:col_dim])
            Ts.writes(reg[0:AB_LANES, 0:AB_LOCAL])
            for ax0, ax1 in T.grid(row_dim, col_dim):
                with Ts.sblock("memory_reg"):
                    v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                    w = index_map(v0, v1)
                    Ts.reads(memory[v0, v1])
                    Ts.writes(reg[w[0], w[1]])
                    reg[w[0], w[1]] = memory[v0, v1]

    s0 = T.dynamic("s0", "int32")
    s1 = T.dynamic("s1", "int32")
    impl_reg_off = T.dynamic("wmma_load_impl_reg_offset", "int32")
    impl_mem_off = T.dynamic("wmma_load_impl_memory_offset", "int32")

    @Ts.function
    def load_impl(
        reg: T.Tensor(
            (1, AB_LOCAL),
            dtype,
            align=32,
            scope="local",
            layout=T.TileLayout(T.S[(1, AB_LOCAL)] + (impl_reg_off)),
        ),
        memory: T.Tensor(
            memory_shape,
            dtype,
            align=32,
            scope=scope,
            layout=T.TileLayout(T.S[memory_shape : [s0, s1]] + (impl_mem_off)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(memory[0:row_dim, 0:col_dim])
            Ts.writes(reg[0:1, 0:AB_LOCAL])
            for local_id in T.serial(0, AB_LOCAL):
                tx = T.launch_thread("threadIdx.x", WARP_SIZE)
                idx = reverse(tx, local_id)
                reg[0, local_id] = memory[idx[0], idx[1]]

    return load_desc, load_impl


def get_wmma_intrin(in_dtype="float16", out_dtype="float32"):
    """D = A x B + C, 16x16x16, wave32. f16 x f16 -> f32 (v_wmma_f32_16x16x16_f16)."""
    assert in_dtype == "float16" and out_dtype == "float32", "only f16 -> f32 for now"
    intrin = "llvm.amdgcn.wmma.f32.16x16x16.f16"
    d_a = T.dynamic("wmma_desc_a_offset", "int32")
    d_b = T.dynamic("wmma_desc_b_offset", "int32")
    d_c = T.dynamic("wmma_desc_c_offset", "int32")

    @Ts.function
    def wmma_desc(
        A: T.Tensor(
            (AB_LANES, AB_LOCAL),
            in_dtype,
            scope="local",
            layout=T.TileLayout(T.S[(AB_LANES, AB_LOCAL)] + (d_a)),
        ),
        B: T.Tensor(
            (AB_LANES, AB_LOCAL),
            in_dtype,
            scope="local",
            layout=T.TileLayout(T.S[(AB_LANES, AB_LOCAL)] + (d_b)),
        ),
        C: T.Tensor(
            (WARP_SIZE, C_LOCAL),
            out_dtype,
            scope="local",
            layout=T.TileLayout(T.S[(WARP_SIZE, C_LOCAL)] + (d_c)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(
                C[0:WARP_SIZE, 0:C_LOCAL], A[0:AB_LANES, 0:AB_LOCAL], B[0:AB_LANES, 0:AB_LOCAL]
            )
            Ts.writes(C[0:WARP_SIZE, 0:C_LOCAL])
            for i, j, k in T.grid(M_DIM, N_DIM, K_DIM):
                with Ts.sblock("C"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    c = shared_16x16_to_local_32x8_layout_C(vi, vj)
                    a = shared_16x16_to_local_32x16_layout_A(vi, vk)
                    b = shared_16x16_to_local_32x16_layout_B(vk, vj)
                    Ts.reads(C[c[0], c[1]], A[a[0], a[1]], B[b[0], b[1]])
                    Ts.writes(C[c[0], c[1]])
                    C[c[0], c[1]] += T.Cast(out_dtype, A[a[0], a[1]]) * T.Cast(
                        out_dtype, B[b[0], b[1]]
                    )

    i_a = T.dynamic("wmma_impl_a_offset", "int32")
    i_b = T.dynamic("wmma_impl_b_offset", "int32")
    i_c = T.dynamic("wmma_impl_c_offset", "int32")

    @Ts.function
    def wmma_impl(
        A: T.Tensor(
            (1, AB_LOCAL), in_dtype, scope="local", layout=T.TileLayout(T.S[(1, AB_LOCAL)] + (i_a))
        ),
        B: T.Tensor(
            (1, AB_LOCAL), in_dtype, scope="local", layout=T.TileLayout(T.S[(1, AB_LOCAL)] + (i_b))
        ),
        C: T.Tensor(
            (1, C_LOCAL), out_dtype, scope="local", layout=T.TileLayout(T.S[(1, C_LOCAL)] + (i_c))
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:1, 0:AB_LOCAL], B[0:1, 0:AB_LOCAL], C[0:1, 0:C_LOCAL])
            Ts.writes(C[0:1, 0:C_LOCAL])
            _tx = T.launch_thread("threadIdx.x", WARP_SIZE)
            C[0, T.ramp(0, 1, C_LOCAL)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id(intrin),
                A[0, T.ramp(0, 1, AB_LOCAL)],
                B[0, T.ramp(0, 1, AB_LOCAL)],
                C[0, T.ramp(0, 1, C_LOCAL)],
                ty=f"{out_dtype}x{C_LOCAL}",
            )

    return wmma_desc, wmma_impl


def get_wmma_store_intrin(dtype="float32", scope="global"):
    @Ts.function
    def store_desc(
        C_warp: T.Tensor([WARP_SIZE, C_LOCAL], dtype=dtype, scope="local"),
        C: T.Tensor([M_DIM, N_DIM], dtype=dtype, scope=scope),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C_warp[0:WARP_SIZE, 0:C_LOCAL])
            Ts.writes(C[0:M_DIM, 0:N_DIM])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with Ts.sblock("C_warp"):
                    v0, v1 = Ts.axis.remap("SS", [i0, i1])
                    w = shared_16x16_to_local_32x8_layout_C(v0, v1)
                    Ts.reads(C_warp[w[0], w[1]])
                    Ts.writes(C[v0, v1])
                    C[v0, v1] = C_warp[w[0], w[1]]

    s0 = T.dynamic("s0", "int32")
    s1 = T.dynamic("s1", "int32")
    o_w = T.dynamic("wmma_store_impl_c_warp_offset", "int32")
    o_c = T.dynamic("wmma_store_impl_c_offset", "int32")

    @Ts.function
    def store_impl(
        C_warp: T.Tensor(
            [1, C_LOCAL], dtype=dtype, scope="local", layout=T.TileLayout(T.S[[1, C_LOCAL]] + (o_w))
        ),
        C: T.Tensor(
            [M_DIM, N_DIM],
            dtype=dtype,
            scope=scope,
            layout=T.TileLayout(T.S[[M_DIM, N_DIM] : [s0, s1]] + (o_c)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C_warp[0:1, 0:C_LOCAL])
            Ts.writes(C[0:M_DIM, 0:N_DIM])
            tx = T.launch_thread("threadIdx.x", WARP_SIZE)
            for i in range(C_LOCAL):
                C[2 * i + tx // 16, tx % 16] = C_warp[0, i]

    return store_desc, store_impl


ROCM_WMMA_FILL_16x16_F32_INTRIN = "rocm_wmma_fill_16x16_f32"
TensorIntrin.register(ROCM_WMMA_FILL_16x16_F32_INTRIN, *get_wmma_fill_intrin("float32"))
ROCM_WMMA_LOAD_16x16_A_SHARED_F16_INTRIN = "rocm_wmma_load_16x16_a_shared_f16"
TensorIntrin.register(
    ROCM_WMMA_LOAD_16x16_A_SHARED_F16_INTRIN, *get_wmma_load_intrin("float16", "shared")
)
ROCM_WMMA_LOAD_16x16_B_SHARED_F16_INTRIN = "rocm_wmma_load_16x16_b_shared_f16"
TensorIntrin.register(
    ROCM_WMMA_LOAD_16x16_B_SHARED_F16_INTRIN, *get_wmma_load_intrin("float16", "shared", is_b=True)
)
ROCM_WMMA_LOAD_16x16_B_TRANS_SHARED_F16_INTRIN = "rocm_wmma_load_16x16_b_trans_shared_f16"
TensorIntrin.register(
    ROCM_WMMA_LOAD_16x16_B_TRANS_SHARED_F16_INTRIN,
    *get_wmma_load_intrin("float16", "shared", is_b=True, transposed=True),
)
ROCM_WMMA_F16F16F32_INTRIN = "rocm_wmma_f16f16f32"
TensorIntrin.register(ROCM_WMMA_F16F16F32_INTRIN, *get_wmma_intrin("float16", "float32"))
ROCM_WMMA_STORE_16x16_F32_INTRIN = "rocm_wmma_store_16x16_f32"
TensorIntrin.register(ROCM_WMMA_STORE_16x16_F32_INTRIN, *get_wmma_store_intrin("float32", "global"))
ROCM_WMMA_STORE_16x16_F32_SHARED_INTRIN = "rocm_wmma_store_16x16_f32_shared"
TensorIntrin.register(
    ROCM_WMMA_STORE_16x16_F32_SHARED_INTRIN, *get_wmma_store_intrin("float32", "shared")
)
