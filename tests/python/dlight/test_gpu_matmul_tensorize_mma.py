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
import pytest

import tvm.testing
from tvm import dlight as dl
from tvm.script import tir as T
from tvm.target import Target


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    @pytest.fixture
    def transform(self):
        def transform(mod):
            with Target("nvidia/nvidia-a100"):
                return dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)

        return transform


class TestNTMatmulMixedPrecision(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(128), T.int64(128)), "float16")
        B = T.match_buffer(p_B, (T.int64(128), T.int64(128)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(128), T.int64(128)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(128), T.int64(128)))
        for i0, i1, i2, k in T.grid(b, T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_i1, v_k]) * T.Cast("float32", B[v_i2, v_k])
        for i0, i1, i2 in T.grid(b, T.int64(128), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def expected(p_A: T.handle, B: T.Buffer((T.int64(128), T.int64(128)), "float16"), p_O: T.handle):
        T.func_attr({"tir.is_scheduled": 1})
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(128), T.int64(128)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(128), T.int64(128)), "float16")
        # with T.block("root"):
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(1), b * T.int64(128), T.int64(128)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), b * T.int64(8), T.int64(8), T.int64(32), T.int64(8)), "float16", scope="warp")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(128)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(32), T.int64(8)), "float16", scope="warp")
        var_matmul_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), b * T.int64(128), T.int64(128)), scope="shared.dyn")
        var_matmul_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), b * T.int64(8), T.int64(8), T.int64(32), T.int64(8)), scope="warp")
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding(b, thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(b * T.int64(8), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(8), ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            with T.block("matmul_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                C_warp = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    T.mma_fill("float32", 8, C_warp.data, C_warp.elem_offset)
            for ax3_0_0 in T.serial(T.int64(4), annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 3]}):
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("A_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(b * T.int64(128), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(A[v1 // T.int64(128), v1 % T.int64(128), v2])
                                        T.writes(A_reindex_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_A"})
                                        A_reindex_shared_dyn[v0, v1, v2] = A[v1 // T.int64(128), v1 % T.int64(128), v2]
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(B[v1, v2])
                                        T.writes(B_reindex_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_B"})
                                        B_reindex_shared_dyn[v0, v1, v2] = B[v1, v2]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("A_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(8) * b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(8), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                    T.reads(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_A"})
                                    with T.block("A_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * (tx % T.int64(16)) + T.int64(8) * (tx // T.int64(16)))
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(8), ax2_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(8), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_B"})
                                    with T.block("B_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.int64(8) * (tx // T.int64(16)) + shared.strides[0] * (tx % T.int64(8)) + T.int64(8) * (tx % T.int64(16) // T.int64(8)))
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(b * T.int64(8), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(8), ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(8), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("matmul_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        A_1 = T.match_buffer(A_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        B_1 = T.match_buffer(B_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        C = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B_1.data, B_1.elem_offset + tx * T.int64(8), C.data, C.elem_offset + tx * T.int64(8), T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B_1.data, B_1.elem_offset + tx * T.int64(8) + T.int64(4), C.data, C.elem_offset + tx * T.int64(8) + T.int64(4), T.bool(False))
            for ax0 in range(T.int64(1)):
                for ax1_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1_1, ax2_1 in T.grid(T.int64(4), T.int64(4)):
                            with T.block("var_matmul_intermediate_reindex_shared.dyn_warp_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(8) * b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                v2_o = T.axis.spatial(T.int64(8), ax2_0 * T.int64(4) + ax2_1)
                                T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                T.writes(var_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.block_attr({"permuted_layout": "l2s_C"})
                                with T.block("var_matmul_intermediate_reindex_shared.dyn_warp_o"):
                                    v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(var_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                    C = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for local_id in range(T.int64(8)):
                                            C[T.int64(8) * (local_id % T.int64(4) // T.int64(2)) + tx // T.int64(4), T.int64(8) * (local_id // T.int64(4)) + tx % T.int64(4) * T.int64(2) + local_id % T.int64(2)] = C_warp[tx, local_id]
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("var_matmul_intermediate_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(b * T.int64(128), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(var_matmul_intermediate_reindex_shared_dyn[v0, v1, v2])
                                    T.writes(O[v1 // T.int64(128), v1 % T.int64(128), v2])
                                    T.block_attr({"permuted_layout": "s2g_C"})
                                    if v1 // T.int64(128) < b:
                                        O[v1 // T.int64(128), v1 % T.int64(128), v2] = T.Cast("float16", var_matmul_intermediate_reindex_shared_dyn[v0, v1, v2])
    # fmt: on


class TestTNMatmulMixedPrecision(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(128), T.int64(128)), "float16")
        B = T.match_buffer(p_B, (T.int64(128), T.int64(128)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(128), T.int64(128)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(128), T.int64(128)))
        for i0, i1, i2, k in T.grid(b, T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_k, v_i1]) * T.Cast("float32", B[v_k, v_i2])
        for i0, i1, i2 in T.grid(b, T.int64(128), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def expected(p_A: T.handle, B: T.Buffer((T.int64(128), T.int64(128)), "float16"), p_O: T.handle):
        T.func_attr({"tir.is_scheduled": 1})
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(128), T.int64(128)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(128), T.int64(128)), "float16")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(128), b * T.int64(128)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(8), b * T.int64(8), T.int64(32), T.int64(8)), "float16", scope="warp")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(128)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(32), T.int64(8)), "float16", scope="warp")
        var_matmul_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), b * T.int64(128), T.int64(128)), scope="shared.dyn")
        var_matmul_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), b * T.int64(8), T.int64(8), T.int64(32), T.int64(8)), scope="warp")
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding(b, thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(b * T.int64(8), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(8), ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            with T.block("matmul_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                C_warp = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    T.mma_fill("float32", 8, C_warp.data, C_warp.elem_offset)
            for ax3_0_0 in T.serial(T.int64(4), annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 3]}):
                for ax1_ax0_fused_0 in range(T.int64(4)):
                    for ax1_ax0_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax1_ax0_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax1_ax0_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax1_ax0_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("A_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(b * T.int64(128), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax1_ax0_fused_0 * T.int64(1024) + ax1_ax0_fused_1 * T.int64(512) + ax1_ax0_fused_2 * T.int64(256) + ax1_ax0_fused_3 * T.int64(8) + ax1_ax0_fused_4) % T.int64(128))
                                        v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(32) + (ax1_ax0_fused_0 * T.int64(1024) + ax1_ax0_fused_1 * T.int64(512) + ax1_ax0_fused_2 * T.int64(256) + ax1_ax0_fused_3 * T.int64(8) + ax1_ax0_fused_4) // T.int64(128))
                                        T.reads(A[v1 // T.int64(128), v2, v1 % T.int64(128)])
                                        T.writes(A_reindex_shared_dyn[v0, v2, v1])
                                        T.block_attr({"permuted_layout": "g2s_A"})
                                        A_reindex_shared_dyn[v0, v2, v1] = A[v1 // T.int64(128), v2, v1 % T.int64(128)]
                for ax1_ax0_fused_0 in range(T.int64(4)):
                    for ax1_ax0_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax1_ax0_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax1_ax0_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax1_ax0_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(128), (ax1_ax0_fused_0 * T.int64(1024) + ax1_ax0_fused_1 * T.int64(512) + ax1_ax0_fused_2 * T.int64(256) + ax1_ax0_fused_3 * T.int64(8) + ax1_ax0_fused_4) % T.int64(128))
                                        v2 = T.axis.spatial(T.int64(128), ax3_0_0 * T.int64(32) + (ax1_ax0_fused_0 * T.int64(1024) + ax1_ax0_fused_1 * T.int64(512) + ax1_ax0_fused_2 * T.int64(256) + ax1_ax0_fused_3 * T.int64(8) + ax1_ax0_fused_4) // T.int64(128))
                                        T.reads(B[v2, v1])
                                        T.writes(B_reindex_shared_dyn[v0, v2, v1])
                                        T.block_attr({"permuted_layout": "g2s_B"})
                                        B_reindex_shared_dyn[v0, v2, v1] = B[v2, v1]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("A_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(8), ax3_0_0 * T.int64(2) + ax3_0_1 + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(8) * b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_A"})
                                    with T.block("A_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.int64(8) * (tx // T.int64(16)) + shared.strides[0] * (tx % T.int64(8)) + T.int64(8) * (tx % T.int64(16) // T.int64(8)))
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("B_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(8), ax3_0_0 * T.int64(2) + ax3_0_1 + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(8), ax2_0_2 * T.int64(4) + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_B"})
                                    with T.block("B_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * (tx % T.int64(16)) + T.int64(8) * (tx // T.int64(16)))
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(b * T.int64(8), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(8), ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(8), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[T.int64(0), v3_o, v1_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_reindex_shared_dyn_warp[T.int64(0), v3_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("matmul_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[T.int64(0), v3_o, v1_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_reindex_shared_dyn_warp[T.int64(0), v3_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.writes(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        A_1 = T.match_buffer(A_reindex_shared_dyn_warp[T.int64(0), v3_o, v1_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        B_1 = T.match_buffer(B_reindex_shared_dyn_warp[T.int64(0), v3_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        C = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B_1.data, B_1.elem_offset + tx * T.int64(8), C.data, C.elem_offset + tx * T.int64(8), T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B_1.data, B_1.elem_offset + tx * T.int64(8) + T.int64(4), C.data, C.elem_offset + tx * T.int64(8) + T.int64(4), T.bool(False))
            for ax0 in range(T.int64(1)):
                for ax1_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1_1, ax2_1 in T.grid(T.int64(4), T.int64(4)):
                            with T.block("var_matmul_intermediate_reindex_shared.dyn_warp_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(8) * b, ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                v2_o = T.axis.spatial(T.int64(8), ax2_0 * T.int64(4) + ax2_1)
                                T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                T.writes(var_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.block_attr({"permuted_layout": "l2s_C"})
                                with T.block("var_matmul_intermediate_reindex_shared.dyn_warp_o"):
                                    v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(var_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(var_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                    C = T.match_buffer(var_matmul_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for local_id in range(T.int64(8)):
                                            C[T.int64(8) * (local_id % T.int64(4) // T.int64(2)) + tx // T.int64(4), T.int64(8) * (local_id // T.int64(4)) + tx % T.int64(4) * T.int64(2) + local_id % T.int64(2)] = C_warp[tx, local_id]
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("var_matmul_intermediate_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(b * T.int64(128), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(var_matmul_intermediate_reindex_shared_dyn[v0, v1, v2])
                                    T.writes(O[v1 // T.int64(128), v1 % T.int64(128), v2])
                                    T.block_attr({"permuted_layout": "s2g_C"})
                                    if v1 // T.int64(128) < b:
                                        O[v1 // T.int64(128), v1 % T.int64(128), v2] = T.Cast("float16", var_matmul_intermediate_reindex_shared_dyn[v0, v1, v2])
    # fmt: on


class TestMatmulDecode(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(
        data: T.Buffer((T.int64(4096), T.int64(512)), "uint32"),
        scale: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
        p_A: T.handle,
        p_O: T.handle,
    ):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(512), T.int64(4096)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(512), T.int64(4096)), "float16")
        B_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
        O_intermediate = T.alloc_buffer((b, T.int64(512), T.int64(4096)))
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(data[v_i, v_j // T.int64(8)], scale[v_i, v_j // T.int64(32)])
                T.writes(B_intermediate[v_i, v_j])
                B_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(data[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * scale[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(b, T.int64(512), T.int64(4096), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B_intermediate[v_i2, v_k])
                T.writes(O_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    O_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                O_intermediate[v_i0, v_i1, v_i2] = O_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_i1, v_k]) * T.Cast("float32", B_intermediate[v_i2, v_k])
        for i0, i1, i2 in T.grid(b, T.int64(512), T.int64(4096)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(O_intermediate[v_i0, v_i1, v_i2])
                T.writes(O[v_i0, v_i1, v_i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", O_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def expected(data: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), scale: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_A: T.handle, p_O: T.handle):
        T.func_attr({"tir.is_scheduled": 1})
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(512), T.int64(4096)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(512), T.int64(4096)), "float16")
        # with T.block("root"):
        B_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(1), b * T.int64(512), T.int64(4096)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), b * T.int64(32), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        B_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), "float16", scope="shared.dyn")
        B_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        O_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), b * T.int64(512), T.int64(4096)), scope="shared.dyn")
        O_intermediate_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), b * T.int64(32), T.int64(256), T.int64(32), T.int64(8)), scope="warp")
        for i_j_fused_0 in T.thread_binding(T.int64(16384), thread="blockIdx.x"):
            for i_j_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for i_j_fused_2 in T.unroll(T.int64(2)):
                    for i_j_fused_3 in T.vectorized(T.int64(4)):
                        with T.block("decode"):
                            v_i = T.axis.spatial(T.int64(4096), (i_j_fused_0 * T.int64(1024) + i_j_fused_1 * T.int64(8) + i_j_fused_2 * T.int64(4) + i_j_fused_3) // T.int64(4096))
                            v_j = T.axis.spatial(T.int64(4096), (i_j_fused_0 * T.int64(1024) + i_j_fused_1 * T.int64(8) + i_j_fused_2 * T.int64(4) + i_j_fused_3) % T.int64(4096))
                            T.reads(data[v_i, v_j // T.int64(8)], scale[v_i, v_j // T.int64(32)])
                            T.writes(B_intermediate[v_i, v_j])
                            B_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(data[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * scale[v_i, v_j // T.int64(32)]
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding(b * T.int64(128), thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("NT_matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(b * T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            with T.block("NT_matmul_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                C_warp = T.match_buffer(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    T.mma_fill("float32", 8, C_warp.data, C_warp.elem_offset)
            for ax3_0_0 in T.serial(T.int64(128), annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 3]}):
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("A_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(b * T.int64(512), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(A[v1 // T.int64(512), v1 % T.int64(512), v2])
                                        T.writes(A_reindex_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_A"})
                                        A_reindex_shared_dyn[v0, v1, v2] = A[v1 // T.int64(512), v1 % T.int64(512), v2]
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("B_intermediate_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(4096), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(B_intermediate[v1, v2])
                                        T.writes(B_intermediate_reindex_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_B"})
                                        B_intermediate_reindex_shared_dyn[v0, v1, v2] = B_intermediate[v1, v2]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("A_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(8) * (b * T.int64(4)), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                    T.reads(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_A"})
                                    with T.block("A_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(A_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(A_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * (tx % T.int64(16)) + T.int64(8) * (tx // T.int64(16)))
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("B_intermediate_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                    T.reads(B_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_B"})
                                    with T.block("B_intermediate_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(B_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(B_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(B_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(B_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.int64(8) * (tx // T.int64(16)) + shared.strides[0] * (tx % T.int64(8)) + T.int64(8) * (tx % T.int64(16) // T.int64(8)))
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("NT_matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(b * T.int64(32), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_intermediate_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("NT_matmul_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_intermediate_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.writes(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        A_1 = T.match_buffer(A_reindex_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        B = T.match_buffer(B_intermediate_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        C = T.match_buffer(O_intermediate_reindex_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8), C.data, C.elem_offset + tx * T.int64(8), T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B.data, B.elem_offset + tx * T.int64(8) + T.int64(4), C.data, C.elem_offset + tx * T.int64(8) + T.int64(4), T.bool(False))
            for ax0 in range(T.int64(1)):
                for ax1_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1_1, ax2_1 in T.grid(T.int64(4), T.int64(4)):
                            with T.block("O_intermediate_reindex_shared.dyn_warp_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(8) * (b * T.int64(4)), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0 * T.int64(4) + ax2_1)
                                T.reads(O_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                T.writes(O_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.block_attr({"permuted_layout": "l2s_C"})
                                with T.block("O_intermediate_reindex_shared.dyn_warp_o"):
                                    v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(O_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(O_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(O_intermediate_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                    C = T.match_buffer(O_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for local_id in range(T.int64(8)):
                                            C[T.int64(8) * (local_id % T.int64(4) // T.int64(2)) + tx // T.int64(4), T.int64(8) * (local_id // T.int64(4)) + tx % T.int64(4) * T.int64(2) + local_id % T.int64(2)] = C_warp[tx, local_id]
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("O_intermediate_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(b * T.int64(512), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(4096), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(O_intermediate_reindex_shared_dyn[v0, v1, v2])
                                    T.writes(O[v1 // T.int64(512), v1 % T.int64(512), v2])
                                    T.block_attr({"permuted_layout": "s2g_C"})
                                    if v1 // T.int64(512) < b:
                                        O[v1 // T.int64(512), v1 % T.int64(512), v2] = T.Cast("float16", O_intermediate_reindex_shared_dyn[v0, v1, v2])
    # fmt: on


class TestMatmulEpilogue(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(
        B: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        p_A: T.handle,
        p_add: T.handle,
        p_add1: T.handle,
        p_O: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (n, T.int64(4096)), "float16")
        add = T.match_buffer(p_add, (n, T.int64(4096)), "float16")
        add1 = T.match_buffer(p_add1, (n, T.int64(4096)), "float16")
        O = T.match_buffer(p_O, (n, T.int64(4096)), "float32")
        O_intermediate = T.alloc_buffer((n, T.int64(4096)))
        O_intermediate1 = T.alloc_buffer((n, T.int64(4096)), "float16")
        O_intermediate2 = T.alloc_buffer((n, T.int64(4096)), "float16")
        O_intermediate3 = T.alloc_buffer((n, T.int64(4096)), "float16")
        for i0, i1, k in T.grid(n, T.int64(4096), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                with T.init():
                    O_intermediate[v_i0, v_i1] = T.float32(0)
                O_intermediate[v_i0, v_i1] = O_intermediate[v_i0, v_i1] + T.Cast("float32", A[v_i0, v_k]) * T.Cast("float32", B[v_i1, v_k])
        for i0, i1 in T.grid(n, T.int64(4096)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                O_intermediate1[v_i0, v_i1] = T.Cast("float16", O_intermediate[v_i0, v_i1])
        for ax0, ax1 in T.grid(n, T.int64(4096)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                O_intermediate2[v_ax0, v_ax1] = O_intermediate1[v_ax0, v_ax1] + add[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(n, T.int64(4096)):
            with T.block("T_add_1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                O_intermediate3[v_ax0, v_ax1] = add1[v_ax0, v_ax1] + O_intermediate2[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(n, T.int64(4096)):
            with T.block("T_cast"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                O[v_ax0, v_ax1] = T.Cast("float32", O_intermediate3[v_ax0, v_ax1])

    @T.prim_func
    def expected(B: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), p_A: T.handle, p_add: T.handle, p_add1: T.handle, p_O: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (n, T.int64(4096)), "float16")
        add = T.match_buffer(p_add, (n, T.int64(4096)), "float16")
        add1 = T.match_buffer(p_add1, (n, T.int64(4096)), "float16")
        O = T.match_buffer(p_O, (n, T.int64(4096)))
        # with T.block("root"):
        A_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(4096)), "float16", scope="shared.dyn")
        A_reindex_pad_shared_dyn_warp = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(8), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn_warp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(256), T.int64(32), T.int64(8)), "float16", scope="warp")
        O_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(4096)), scope="shared.dyn")
        O_intermediate_reindex_pad_shared_dyn_warp = T.alloc_buffer((T.int64(1), (n + T.int64(127)) // T.int64(128) * T.int64(8), T.int64(256), T.int64(32), T.int64(8)), scope="warp")
        for ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused in T.thread_binding((n + T.int64(127)) // T.int64(128) * T.int64(32), thread="blockIdx.x"):
            for ax1_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                for ax2_0_2_init in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(4), T.int64(4)):
                        with T.block("NT_matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2_init * T.int64(4) + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2_init * T.int64(4) + ax2_0_3_init)
                            T.reads()
                            T.writes(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            with T.block("NT_matmul_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                C_warp = T.match_buffer(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    T.mma_fill("float32", 8, C_warp.data, C_warp.elem_offset)
            for ax3_0_0 in T.serial(T.int64(128), annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 3]}):
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("A_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(A[v1, v2])
                                        T.writes(A_reindex_pad_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_A"})
                                        A_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < n, A[v1, v2], T.float16(0))
                for ax0_ax1_fused_0 in range(T.int64(4)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                            for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(4096), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(32))
                                        v2 = T.axis.spatial(T.int64(4096), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(32))
                                        T.reads(B[v1, v2])
                                        T.writes(B_reindex_shared_dyn[v0, v1, v2])
                                        T.block_attr({"permuted_layout": "g2s_B"})
                                        B_reindex_shared_dyn[v0, v1, v2] = B[v1, v2]
                for ax1_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("A_reindex_pad_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                    T.reads(A_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_pad_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_A"})
                                    with T.block("A_reindex_pad_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(A_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(A_reindex_pad_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(A_reindex_pad_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(A_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * (tx % T.int64(16)) + T.int64(8) * (tx // T.int64(16)))
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_warp_o"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2 * T.int64(4) + ax0_0)
                                    v2_o = T.axis.spatial(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.block_attr({"permuted_layout": "s2l_B"})
                                    with T.block("B_reindex_shared.dyn_warp_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        T.reads(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        warp = T.match_buffer(B_reindex_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        shared = T.match_buffer(B_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("shared_s0", "shared_s1"), scope="shared.dyn", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + T.int64(8) * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * T.int64(16), 1), shared.strides[0] * T.int64(8) * (tx // T.int64(16)) + shared.strides[0] * (tx % T.int64(8)) + T.int64(8) * (tx % T.int64(16) // T.int64(8)))
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(4), T.int64(4)):
                                with T.block("NT_matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1_o = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(8), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0_2 * T.int64(4) + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0_2 * T.int64(4) + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(256), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("NT_matmul_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], A_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], B_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.writes(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        A_1 = T.match_buffer(A_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        B_1 = T.match_buffer(B_reindex_shared_dyn_warp[T.int64(0), v2_o, v3_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), "float16", scope="warp", offset_factor=16)
                                        C = T.match_buffer(O_intermediate_reindex_pad_shared_dyn_warp[T.int64(0), v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=16)
                                        for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B_1.data, B_1.elem_offset + tx * T.int64(8), C.data, C.elem_offset + tx * T.int64(8), T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_1.data, A_1.elem_offset + tx * T.int64(8), B_1.data, B_1.elem_offset + tx * T.int64(8) + T.int64(4), C.data, C.elem_offset + tx * T.int64(8) + T.int64(4), T.bool(False))
            for ax0 in range(T.int64(1)):
                for ax1_0 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax2_0 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax1_1, ax2_1 in T.grid(T.int64(4), T.int64(4)):
                            with T.block("O_intermediate_reindex_pad_shared.dyn_warp_o"):
                                v0_o = T.axis.spatial(T.int64(1), ax0)
                                v1_o = T.axis.spatial(T.int64(8) * ((n + T.int64(127)) // T.int64(128)), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(8) + ax1_0 * T.int64(4) + ax1_1)
                                v2_o = T.axis.spatial(T.int64(256), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(8) + ax2_0 * T.int64(4) + ax2_1)
                                T.reads(O_intermediate_reindex_pad_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                T.writes(O_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                T.block_attr({"permuted_layout": "l2s_C"})
                                with T.block("O_intermediate_reindex_pad_shared.dyn_warp_o"):
                                    v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(O_intermediate_reindex_pad_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    T.writes(O_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    C_warp = T.match_buffer(O_intermediate_reindex_pad_shared_dyn_warp[v0_o, v1_o, v2_o, T.int64(0):T.int64(32), T.int64(0):T.int64(8)], (T.int64(32), T.int64(8)), scope="warp", offset_factor=1)
                                    C = T.match_buffer(O_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=1)
                                    for tx in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        for local_id in range(T.int64(8)):
                                            C[T.int64(8) * (local_id % T.int64(4) // T.int64(2)) + tx // T.int64(4), T.int64(8) * (local_id // T.int64(4)) + tx % T.int64(4) * T.int64(2) + local_id % T.int64(2)] = C_warp[tx, local_id]
            for ax0_ax1_fused_0 in range(T.int64(16)):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(2), thread="threadIdx.z"):
                    for ax0_ax1_fused_2 in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                        for ax0_ax1_fused_3 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_4 in T.vectorized(T.int64(8)):
                                with T.block("O_intermediate_reindex_pad_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial((n + T.int64(127)) // T.int64(128) * T.int64(128), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused // T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) // T.int64(128))
                                    v2 = T.axis.spatial(T.int64(4096), ax0_ax1_0_0_ax2_0_0_ax1_0_1_ax2_0_1_fused % T.int64(32) * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(512) + ax0_ax1_fused_2 * T.int64(256) + ax0_ax1_fused_3 * T.int64(8) + ax0_ax1_fused_4) % T.int64(128))
                                    T.reads(add1[v1, v2], O_intermediate_reindex_pad_shared_dyn[v0, v1, v2], add[v1, v2])
                                    T.writes(O[v1, v2])
                                    T.block_attr({"permuted_layout": "s2g_C"})
                                    if v1 < n:
                                        O[v1, v2] = T.Cast("float32", add1[v1, v2] + (T.Cast("float16", O_intermediate_reindex_pad_shared_dyn[v0, v1, v2]) + add[v1, v2]))
    # fmt: on


if __name__ == "__main__":
    tvm.testing.main()
