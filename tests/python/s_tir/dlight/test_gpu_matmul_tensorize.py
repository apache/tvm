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
# pylint: disable=missing-docstring, unused-variable, invalid-name
# ruff: noqa: E501

import tvm
import tvm.testing
from tvm.s_tir import dlight as dl
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.target import Target


def test_matmul_tensorize_too_small():
    # fmt: off
    m = T.dynamic("m", "int32")

    @Ts.function(private=True)
    def before(X: T.Tensor((m, 256), 'float16'), W: T.Tensor((15, 256), "float16"), compute: T.Tensor((m, 15))):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        for i, j, k in T.grid(m, 15, 256):
            with Ts.sblock("compute"):
                v_i, v_j, v_k = Ts.axis.remap("SSR", [i, j, k])
                Ts.reads(X[v_i, v_k], W[v_j, v_k])
                Ts.writes(compute[v_i, v_j])
                with Ts.init():
                    compute[v_i, v_j] = T.float32(0)
                compute[v_i, v_j] = compute[v_i, v_j] + T.Cast("float32", X[v_i, v_k]) * T.Cast("float32", W[v_j, v_k])

    m = T.dynamic("m", "int32")

    @Ts.function(private=True)
    def expected(X: T.Tensor((m, 256), 'float16'), W: T.Tensor((15, 256), "float16"), compute: T.Tensor((m, 15))):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        compute_reindex_pad_local = Ts.sblock_alloc_buffer((1, (m + 31) // 32 * 32, 64), scope="local")
        X_reindex_pad_shared = Ts.sblock_alloc_buffer((1, (m + 31) // 32 * 32, 256), "float16", scope="shared")
        W_reindex_pad_shared = Ts.sblock_alloc_buffer((1, 64, 256), "float16", scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax1_0 in T.thread_binding((m + 31) // 32, thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(1, thread="vthread.y"):
                    for ax1_1 in T.thread_binding(1, thread="vthread.x"):
                        for ax2_2 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(8, thread="threadIdx.x", annotations={"auto_unroll_max_step": 256, "unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(4, 2):
                                    for ax2_3_1_init in T.vectorized(2):
                                        with Ts.sblock("compute_init"):
                                            v0 = Ts.axis.spatial(1, 0)
                                            v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3_init)
                                            v2 = Ts.axis.spatial(64, ax2_1 * 64 + ax2_2 * 4 + ax2_3_0_init * 2 + ax2_3_1_init)
                                            Ts.reads()
                                            Ts.writes(compute_reindex_pad_local[0, v1, v2])
                                            compute_reindex_pad_local[0, v1, v2] = T.float32(0)
                                for ax3_0 in range(16):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(2):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with Ts.sblock("X_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(1, 0)
                                                        v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = Ts.axis.spatial(256, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        Ts.reads(X[v1, v2])
                                                        Ts.writes(X_reindex_pad_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        X_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, X[v1, v2], T.float16(0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(4):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with Ts.sblock("W_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(1, 0)
                                                        v1 = Ts.axis.spatial(64, (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = Ts.axis.spatial(256, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        Ts.reads(W[v1, v2])
                                                        Ts.writes(W_reindex_pad_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        W_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < 15, W[v1, v2], T.float16(0))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(16, 4, 2):
                                        for ax2_3_1 in T.vectorized(2):
                                            with Ts.sblock("compute_update"):
                                                v0 = Ts.axis.spatial(1, 0)
                                                v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3)
                                                v2 = Ts.axis.spatial(64, ax2_1 * 64 + ax2_2 * 4 + ax2_3_0 * 2 + ax2_3_1)
                                                v3 = Ts.axis.reduce(256, ax3_0 * 16 + ax3_1)
                                                Ts.reads(compute_reindex_pad_local[0, v1, v2], X_reindex_pad_shared[0, v1, v3], W_reindex_pad_shared[0, v2, v3])
                                                Ts.writes(compute_reindex_pad_local[0, v1, v2])
                                                compute_reindex_pad_local[0, v1, v2] = compute_reindex_pad_local[0, v1, v2] + T.Cast("float32", X_reindex_pad_shared[0, v1, v3]) * T.Cast("float32", W_reindex_pad_shared[0, v2, v3])
                                for ax0, ax1, ax2_0 in T.grid(1, 4, 2):
                                    for ax2_1_1 in T.vectorized(2):
                                        with Ts.sblock("compute_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(1, ax0)
                                            v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_2 * 4 + ax1)
                                            v2 = Ts.axis.spatial(64, ax2_2 * 4 + ax2_0 * 2 + ax2_1_1)
                                            Ts.where(ax1_0 * 32 + ax1_2 * 4 + ax1 < m and ax2_2 * 4 + ax2_0 * 2 + ax2_1_1 < 15)
                                            Ts.reads(compute_reindex_pad_local[v0, v1, v2])
                                            Ts.writes(compute[v1, v2])
                                            compute[v1, v2] = compute_reindex_pad_local[v0, v1, v2]
    # fmt: on

    mod = tvm.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-2080-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_metal():
    # fmt: off
    batch_size = T.dynamic("batch_size", "int32")

    @Ts.function(private=True)
    def before(
        A: T.Tensor((batch_size, 1, 4096), 'float16'),
        B: T.Tensor((28672, 4096), "float16"),
        C: T.Tensor((batch_size, 1, 28672), 'float16'),
    ):

        for i0, i1, i2, k in T.grid(batch_size, 1, 28672, 4096):
            with Ts.sblock("C"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.writes(C[v_i0, v_i1, v_i2])
                with Ts.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0)
                C[v_i0, v_i1, v_i2] += A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    batch_size = T.dynamic("batch_size", "int32")
    A_s0 = T.dynamic("A_s0", "int32")
    A_s1 = T.dynamic("A_s1", "int32")
    A_4_s0 = T.dynamic("A_4_s0", "int32")
    A_4_s1 = T.dynamic("A_4_s1", "int32")
    C_3_s0 = T.dynamic("C_3_s0", "int32")
    C_3_s1 = T.dynamic("C_3_s1", "int32")
    A_1_s0 = T.dynamic("A_1_s0", "int32")
    A_1_s1 = T.dynamic("A_1_s1", "int32")
    C_s0 = T.dynamic("C_s0", "int32")
    C_s1 = T.dynamic("C_s1", "int32")
    A_2_s0 = T.dynamic("A_2_s0", "int32")
    A_2_s1 = T.dynamic("A_2_s1", "int32")
    C_1_s0 = T.dynamic("C_1_s0", "int32")
    C_1_s1 = T.dynamic("C_1_s1", "int32")
    A_3_s0 = T.dynamic("A_3_s0", "int32")
    A_3_s1 = T.dynamic("A_3_s1", "int32")
    B_s0 = T.dynamic("B_s0", "int32")
    B_s1 = T.dynamic("B_s1", "int32")
    C_2_s0 = T.dynamic("C_2_s0", "int32")
    C_2_s1 = T.dynamic("C_2_s1", "int32")

    @Ts.function(private=True)
    def expected(A: T.Tensor((batch_size, 1, 4096), 'float16'), B: T.Tensor((28672, 4096), "float16"), C: T.Tensor((batch_size, 1, 28672), 'float16')):
        T.func_attr({"tirx.is_scheduled": True})

        # with Ts.sblock("root"):
        A_reindex_pad_shared = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 4096), "float16", scope="shared")
        B_reindex_shared = Ts.sblock_alloc_buffer((1, 28672, 4096), "float16", scope="shared")
        A_reindex_pad_shared_metal_simdgroup = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 4096), "float16", scope="metal.simdgroup")
        B_reindex_shared_metal_simdgroup = Ts.sblock_alloc_buffer((1, 4096, 28672), "float16", scope="metal.simdgroup")
        C_reindex_pad_metal_simdgroup = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 28672), "float16", scope="metal.simdgroup")
        C_reindex_pad_shared = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 28672), "float16", scope="shared")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((batch_size + 15) // 16, thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(448, thread="blockIdx.y"):
                    for ax1_1 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_1 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_2_init, ax2_2_init, ax1_3_init_0, ax2_3_init_0 in T.grid(2, 2, 1, 1):
                                with Ts.sblock("C_init_o"):
                                    v0_o = Ts.axis.spatial(1, ax0)
                                    v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax1_1 * 2 + ax1_2_init + ax1_3_init_0)
                                    v2_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax2_2_init + ax2_3_init_0)
                                    Ts.reads()
                                    Ts.writes(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                    A_1 = Ts.match_buffer(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_s0, A_s1), scope="metal.simdgroup", offset_factor=1)
                                    T.metal.make_filled_simdgroup_matrix(A_1.data, A_1.elem_offset // A_1.strides[0] // 8 * (A_1.strides[0] // 8) + A_1.elem_offset % A_1.strides[0] // 8, T.float32(0), 8, 8)
                            for ax3_0 in range(128):
                                for ax0_1, ax1_ax2_fused_0 in T.grid(1, 1):
                                    for ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax1_ax2_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                            for ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax1_ax2_fused_4 in T.vectorized(4):
                                                    with Ts.sblock("A_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(1, ax0_1)
                                                        v1 = Ts.axis.spatial((batch_size + 15) // 16 * 16, ax1_0 * 16 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) // 32)
                                                        v2 = Ts.axis.spatial(4096, ax3_0 * 32 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) % 32)
                                                        Ts.reads(A[v1, 0, v2])
                                                        Ts.writes(A_reindex_pad_shared[v0, v1, v2])
                                                        A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < batch_size, A[v1, 0, v2], T.float16(0))
                                for ax0_1, ax1_ax2_fused_0 in T.grid(1, 4):
                                    for ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax1_ax2_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                            for ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax1_ax2_fused_4 in T.vectorized(4):
                                                    with Ts.sblock("B_reindex_shared"):
                                                        v0 = Ts.axis.spatial(1, ax0_1)
                                                        v1 = Ts.axis.spatial(28672, ax2_0 * 64 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) // 32)
                                                        v2 = Ts.axis.spatial(4096, ax3_0 * 32 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) % 32)
                                                        Ts.reads(B[v1, v2])
                                                        Ts.writes(B_reindex_shared[v0, v1, v2])
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2]
                                for ax3_1 in range(4):
                                    for ax0_0, ax1_0_1 in T.grid(2, 1):
                                        with Ts.sblock("A_reindex_pad_shared_metal.simdgroup_o"):
                                            v0_o = Ts.axis.spatial(1, 0)
                                            v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax0_0)
                                            v2_o = Ts.axis.spatial(512, ax3_0 * 4 + ax3_1 + ax1_0_1)
                                            Ts.reads(A_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            Ts.writes(A_reindex_pad_shared_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            A_1 = Ts.match_buffer(A_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_1_s0, A_1_s1), scope="shared", offset_factor=1)
                                            C_1 = Ts.match_buffer(A_reindex_pad_shared_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(C_s0, C_s1), scope="metal.simdgroup", offset_factor=1)
                                            T.metal.simdgroup_load(C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8, T.access_ptr("float16", A_1.data, A_1.elem_offset, A_1.strides[0] * 8, 1), A_1.strides[0], 8, 8, T.bool(False))
                                    for ax0_0, ax1_0_1 in T.grid(2, 1):
                                        with Ts.sblock("B_reindex_shared_metal.simdgroup_o"):
                                            v0_o = Ts.axis.spatial(1, 0)
                                            v1_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax0_0)
                                            v2_o = Ts.axis.spatial(512, ax3_0 * 4 + ax3_1 + ax1_0_1)
                                            Ts.reads(B_reindex_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            Ts.writes(B_reindex_shared_metal_simdgroup[v0_o, v2_o * 8:v2_o * 8 + 8, v1_o * 8:v1_o * 8 + 8])
                                            A_1 = Ts.match_buffer(B_reindex_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_2_s0, A_2_s1), scope="shared", offset_factor=1)
                                            C_1 = Ts.match_buffer(B_reindex_shared_metal_simdgroup[v0_o, v2_o * 8:v2_o * 8 + 8, v1_o * 8:v1_o * 8 + 8], (8, 8), "float16", strides=(C_1_s0, C_1_s1), scope="metal.simdgroup", offset_factor=1)
                                            T.metal.simdgroup_load(C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8, T.access_ptr("float16", A_1.data, A_1.elem_offset, A_1.strides[0] * 8, 1), A_1.strides[0], 8, 8, T.bool(True))
                                    for ax1_2, ax2_2 in T.grid(2, 2):
                                        with Ts.sblock("C_update_o"):
                                            v0_o = Ts.axis.spatial(1, ax0)
                                            v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax1_1 * 2 + ax1_2)
                                            v2_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax2_2)
                                            v3_o = Ts.axis.reduce(512, ax3_0 * 4 + ax3_1)
                                            Ts.reads(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], A_reindex_pad_shared_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v3_o * 8:v3_o * 8 + 8], B_reindex_shared_metal_simdgroup[0, v3_o * 8:v3_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            Ts.writes(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            A_1 = Ts.match_buffer(A_reindex_pad_shared_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v3_o * 8:v3_o * 8 + 8], (8, 8), "float16", strides=(A_3_s0, A_3_s1), scope="metal.simdgroup", offset_factor=1)
                                            B_1 = Ts.match_buffer(B_reindex_shared_metal_simdgroup[0, v3_o * 8:v3_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(B_s0, B_s1), scope="metal.simdgroup", offset_factor=1)
                                            C_1 = Ts.match_buffer(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(C_2_s0, C_2_s1), scope="metal.simdgroup", offset_factor=1)
                                            T.metal.simdgroup_multiply_accumulate(C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8, A_1.data, A_1.elem_offset // A_1.strides[0] // 8 * (A_1.strides[0] // 8) + A_1.elem_offset % A_1.strides[0] // 8, B_1.data, B_1.elem_offset // B_1.strides[0] // 8 * (B_1.strides[0] // 8) + B_1.elem_offset % B_1.strides[0] // 8, C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8)
                            for ax0_1, ax1_0_1, ax2_0_1 in T.grid(1, 2, 2):
                                with Ts.sblock("C_reindex_pad_metal.simdgroup_o"):
                                    v0_o = Ts.axis.spatial(1, ax0_1)
                                    v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax1_0_1)
                                    v2_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax2_0_1)
                                    Ts.reads(C_reindex_pad_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                    Ts.writes(C_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                    A_1 = Ts.match_buffer(C_reindex_pad_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_4_s0, A_4_s1), scope="metal.simdgroup", offset_factor=1)
                                    C_1 = Ts.match_buffer(C_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(C_3_s0, C_3_s1), scope="shared", offset_factor=1)
                                    T.metal.simdgroup_store(A_1.data, A_1.elem_offset // A_1.strides[0] // 8 * (A_1.strides[0] // 8) + A_1.elem_offset % A_1.strides[0] // 8, T.access_ptr("float16", C_1.data, C_1.elem_offset, C_1.strides[0] * 8, 2), C_1.strides[0], 8, 8, T.bool(False))
                    for ax0_1, ax1_ax2_fused_0 in T.grid(1, 2):
                        for ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_ax2_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax1_ax2_fused_4 in T.vectorized(4):
                                        with Ts.sblock("C_reindex_pad_shared"):
                                            v0 = Ts.axis.spatial(1, ax0_1)
                                            v1 = Ts.axis.spatial((batch_size + 15) // 16 * 16, ax1_0 * 16 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) // 64)
                                            v2 = Ts.axis.spatial(28672, ax2_0 * 64 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) % 64)
                                            Ts.where(ax1_0 * 16 + (((ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1 + ax1_ax2_fused_2) * 32 + ax1_ax2_fused_3) * 4 + ax1_ax2_fused_4) // 64 < batch_size)
                                            Ts.reads(C_reindex_pad_shared[v0, v1, v2])
                                            Ts.writes(C[v1, 0, v2])
                                            C[v1, 0, v2] = C_reindex_pad_shared[v0, v1, v2]
    # fmt: on

    mod = tvm.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_metal_int4_quant():
    # fmt: off
    batch_size = T.dynamic("batch_size", "int32")

    @Ts.function(private=True)
    def before(
        B0: T.Tensor((28672, 512), "uint32"),
        B1: T.Tensor((28672, 128), "float16"),
        A: T.Tensor((batch_size, 1, 4096), 'float16'),
        C: T.Tensor((batch_size, 1, 28672), 'float16')
    ):

        compute = Ts.sblock_alloc_buffer((28672, 4096), "float16")
        B = Ts.sblock_alloc_buffer((28672, 4096), "float16")
        for i0, i1 in T.grid(28672, 4096):
            with Ts.sblock("compute"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(B0[v_i0, v_i1 // 8], T.Cast("uint32", v_i1 % 8 * 4)), T.uint32(15)))
        for i0, i1 in T.grid(28672, 4096):
            with Ts.sblock("dequantize"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                B[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * B1[v_i0, v_i1 // 32]
        for i0, i1, i2, k in T.grid(batch_size, 1, 28672, 4096):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                with Ts.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    batch_size = T.dynamic("batch_size", "int32")
    A_s0 = T.dynamic("A_s0", "int32")
    A_s1 = T.dynamic("A_s1", "int32")
    A_4_s0 = T.dynamic("A_4_s0", "int32")
    A_4_s1 = T.dynamic("A_4_s1", "int32")
    C_3_s0 = T.dynamic("C_3_s0", "int32")
    C_3_s1 = T.dynamic("C_3_s1", "int32")
    A_1_s0 = T.dynamic("A_1_s0", "int32")
    A_1_s1 = T.dynamic("A_1_s1", "int32")
    C_s0 = T.dynamic("C_s0", "int32")
    C_s1 = T.dynamic("C_s1", "int32")
    A_2_s0 = T.dynamic("A_2_s0", "int32")
    A_2_s1 = T.dynamic("A_2_s1", "int32")
    C_1_s0 = T.dynamic("C_1_s0", "int32")
    C_1_s1 = T.dynamic("C_1_s1", "int32")
    A_3_s0 = T.dynamic("A_3_s0", "int32")
    A_3_s1 = T.dynamic("A_3_s1", "int32")
    B_s0 = T.dynamic("B_s0", "int32")
    B_s1 = T.dynamic("B_s1", "int32")
    C_2_s0 = T.dynamic("C_2_s0", "int32")
    C_2_s1 = T.dynamic("C_2_s1", "int32")

    @Ts.function(private=True)
    def expected(B0: T.Tensor((28672, 512), "uint32"), B1: T.Tensor((28672, 128), "float16"), A: T.Tensor((batch_size, 1, 4096), 'float16'), C: T.Tensor((batch_size, 1, 28672), 'float16')):
        T.func_attr({"tirx.is_scheduled": True})

        # with Ts.sblock("root"):
        A_reindex_pad_shared = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 4096), "float16", scope="shared")
        B_reindex_shared = Ts.sblock_alloc_buffer((1, 28672, 4096), "float16", scope="shared")
        A_reindex_pad_shared_metal_simdgroup = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 4096), "float16", scope="metal.simdgroup")
        B_reindex_shared_metal_simdgroup = Ts.sblock_alloc_buffer((1, 4096, 28672), "float16", scope="metal.simdgroup")
        C_reindex_pad_metal_simdgroup = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 28672), "float16", scope="metal.simdgroup")
        C_reindex_pad_shared = Ts.sblock_alloc_buffer((1, (batch_size + 15) // 16 * 16, 28672), "float16", scope="shared")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((batch_size + 15) // 16, thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(448, thread="blockIdx.y"):
                    for ax1_1 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_1 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_2_init, ax2_2_init, ax1_3_init_0, ax2_3_init_0 in T.grid(2, 2, 1, 1):
                                with Ts.sblock("NT_matmul_init_o"):
                                    v0_o = Ts.axis.spatial(1, ax0)
                                    v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax1_1 * 2 + ax1_2_init + ax1_3_init_0)
                                    v2_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax2_2_init + ax2_3_init_0)
                                    Ts.reads()
                                    Ts.writes(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                    A_1 = Ts.match_buffer(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_s0, A_s1), scope="metal.simdgroup", offset_factor=1)
                                    T.metal.make_filled_simdgroup_matrix(A_1.data, A_1.elem_offset // A_1.strides[0] // 8 * (A_1.strides[0] // 8) + A_1.elem_offset % A_1.strides[0] // 8, T.float32(0), 8, 8)
                            for ax3_0 in range(128):
                                for ax0_1, ax1_ax2_fused_0 in T.grid(1, 1):
                                    for ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax1_ax2_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                            for ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax1_ax2_fused_4 in T.vectorized(4):
                                                    with Ts.sblock("A_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(1, ax0_1)
                                                        v1 = Ts.axis.spatial((batch_size + 15) // 16 * 16, ax1_0 * 16 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) // 32)
                                                        v2 = Ts.axis.spatial(4096, ax3_0 * 32 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) % 32)
                                                        Ts.reads(A[v1, 0, v2])
                                                        Ts.writes(A_reindex_pad_shared[v0, v1, v2])
                                                        A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < batch_size, A[v1, 0, v2], T.float16(0))
                                for ax0_1, ax1_ax2_fused_0 in T.grid(1, 4):
                                    for ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax1_ax2_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                            for ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax1_ax2_fused_4 in T.vectorized(4):
                                                    with Ts.sblock("B_reindex_shared"):
                                                        v0 = Ts.axis.spatial(1, ax0_1)
                                                        v1 = Ts.axis.spatial(28672, ax2_0 * 64 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) // 32)
                                                        v2 = Ts.axis.spatial(4096, ax3_0 * 32 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) % 32)
                                                        Ts.reads(B0[v1, v2 // 8], B1[v1, v2 // 32])
                                                        Ts.writes(B_reindex_shared[v0, v1, v2])
                                                        B_reindex_shared[v0, v1, v2] = (T.Cast("float16", T.bitwise_and(T.shift_right(B0[v1, v2 // 8], T.Cast("uint32", v2 % 8 * 4)), T.uint32(15))) - T.float16(7)) * B1[v1, v2 // 32]
                                for ax3_1 in range(4):
                                    for ax0_0, ax1_0_1 in T.grid(2, 1):
                                        with Ts.sblock("A_reindex_pad_shared_metal.simdgroup_o"):
                                            v0_o = Ts.axis.spatial(1, 0)
                                            v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax0_0)
                                            v2_o = Ts.axis.spatial(512, ax3_0 * 4 + ax3_1 + ax1_0_1)
                                            Ts.reads(A_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            Ts.writes(A_reindex_pad_shared_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            A_1 = Ts.match_buffer(A_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_1_s0, A_1_s1), scope="shared", offset_factor=1)
                                            C_1 = Ts.match_buffer(A_reindex_pad_shared_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(C_s0, C_s1), scope="metal.simdgroup", offset_factor=1)
                                            T.metal.simdgroup_load(C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8, T.access_ptr("float16", A_1.data, A_1.elem_offset, A_1.strides[0] * 8, 1), A_1.strides[0], 8, 8, T.bool(False))
                                    for ax0_0, ax1_0_1 in T.grid(2, 1):
                                        with Ts.sblock("B_reindex_shared_metal.simdgroup_o"):
                                            v0_o = Ts.axis.spatial(1, 0)
                                            v1_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax0_0)
                                            v2_o = Ts.axis.spatial(512, ax3_0 * 4 + ax3_1 + ax1_0_1)
                                            Ts.reads(B_reindex_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            Ts.writes(B_reindex_shared_metal_simdgroup[v0_o, v2_o * 8:v2_o * 8 + 8, v1_o * 8:v1_o * 8 + 8])
                                            A_1 = Ts.match_buffer(B_reindex_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_2_s0, A_2_s1), scope="shared", offset_factor=1)
                                            C_1 = Ts.match_buffer(B_reindex_shared_metal_simdgroup[v0_o, v2_o * 8:v2_o * 8 + 8, v1_o * 8:v1_o * 8 + 8], (8, 8), "float16", strides=(C_1_s0, C_1_s1), scope="metal.simdgroup", offset_factor=1)
                                            T.metal.simdgroup_load(C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8, T.access_ptr("float16", A_1.data, A_1.elem_offset, A_1.strides[0] * 8, 1), A_1.strides[0], 8, 8, T.bool(True))
                                    for ax1_2, ax2_2 in T.grid(2, 2):
                                        with Ts.sblock("NT_matmul_update_o"):
                                            v0_o = Ts.axis.spatial(1, ax0)
                                            v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax1_1 * 2 + ax1_2)
                                            v2_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax2_2)
                                            v3_o = Ts.axis.reduce(512, ax3_0 * 4 + ax3_1)
                                            Ts.reads(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], A_reindex_pad_shared_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v3_o * 8:v3_o * 8 + 8], B_reindex_shared_metal_simdgroup[0, v3_o * 8:v3_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            Ts.writes(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                            A_1 = Ts.match_buffer(A_reindex_pad_shared_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v3_o * 8:v3_o * 8 + 8], (8, 8), "float16", strides=(A_3_s0, A_3_s1), scope="metal.simdgroup", offset_factor=1)
                                            B = Ts.match_buffer(B_reindex_shared_metal_simdgroup[0, v3_o * 8:v3_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(B_s0, B_s1), scope="metal.simdgroup", offset_factor=1)
                                            C_1 = Ts.match_buffer(C_reindex_pad_metal_simdgroup[0, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(C_2_s0, C_2_s1), scope="metal.simdgroup", offset_factor=1)
                                            T.metal.simdgroup_multiply_accumulate(C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8, A_1.data, A_1.elem_offset // A_1.strides[0] // 8 * (A_1.strides[0] // 8) + A_1.elem_offset % A_1.strides[0] // 8, B.data, B.elem_offset // B.strides[0] // 8 * (B.strides[0] // 8) + B.elem_offset % B.strides[0] // 8, C_1.data, C_1.elem_offset // C_1.strides[0] // 8 * (C_1.strides[0] // 8) + C_1.elem_offset % C_1.strides[0] // 8)
                            for ax0_1, ax1_0_1, ax2_0_1 in T.grid(1, 2, 2):
                                with Ts.sblock("C_reindex_pad_metal.simdgroup_o"):
                                    v0_o = Ts.axis.spatial(1, ax0_1)
                                    v1_o = Ts.axis.spatial(2 * ((batch_size + 15) // 16), ax1_0 * 2 + ax1_0_1)
                                    v2_o = Ts.axis.spatial(3584, ax2_0 * 8 + ax2_1 * 2 + ax2_0_1)
                                    Ts.reads(C_reindex_pad_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                    Ts.writes(C_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8])
                                    A_1 = Ts.match_buffer(C_reindex_pad_metal_simdgroup[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(A_4_s0, A_4_s1), scope="metal.simdgroup", offset_factor=1)
                                    C_1 = Ts.match_buffer(C_reindex_pad_shared[v0_o, v1_o * 8:v1_o * 8 + 8, v2_o * 8:v2_o * 8 + 8], (8, 8), "float16", strides=(C_3_s0, C_3_s1), scope="shared", offset_factor=1)
                                    T.metal.simdgroup_store(A_1.data, A_1.elem_offset // A_1.strides[0] // 8 * (A_1.strides[0] // 8) + A_1.elem_offset % A_1.strides[0] // 8, T.access_ptr("float16", C_1.data, C_1.elem_offset, C_1.strides[0] * 8, 2), C_1.strides[0], 8, 8, T.bool(False))
                    for ax0_1, ax1_ax2_fused_0 in T.grid(1, 2):
                        for ax1_ax2_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_ax2_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax1_ax2_fused_4 in T.vectorized(4):
                                        with Ts.sblock("C_reindex_pad_shared"):
                                            v0 = Ts.axis.spatial(1, ax0_1)
                                            v1 = Ts.axis.spatial((batch_size + 15) // 16 * 16, ax1_0 * 16 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) // 64)
                                            v2 = Ts.axis.spatial(28672, ax2_0 * 64 + (ax1_ax2_fused_0 * 512 + ax1_ax2_fused_1 * 128 + ax1_ax2_fused_2 * 128 + ax1_ax2_fused_3 * 4 + ax1_ax2_fused_4) % 64)
                                            Ts.where(ax1_0 * 16 + (((ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1 + ax1_ax2_fused_2) * 32 + ax1_ax2_fused_3) * 4 + ax1_ax2_fused_4) // 64 < batch_size)
                                            Ts.reads(C_reindex_pad_shared[v0, v1, v2])
                                            Ts.writes(C[v1, 0, v2])
                                            C[v1, 0, v2] = C_reindex_pad_shared[v0, v1, v2]
    # fmt: on

    mod = tvm.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
