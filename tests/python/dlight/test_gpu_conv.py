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
            with Target("nvidia/geforce-gtx-1080-ti"):
                # Use Matmul rule for Conv for now
                return dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)

        return transform


class TestConv3d(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(
        A: T.Buffer((14308, 3, 2, 14, 14), "float16"),
        W: T.Buffer((1280, 3, 2, 14, 14), "float16"),
        C: T.Buffer((14308, 1280, 1, 1, 1), "float16"),
    ):
        pad_A = T.alloc_buffer((14308, 3, 2, 14, 14), "float16")
        for i0, i1, i2, i3, i4 in T.grid(14308, 3, 2, 14, 14):
            with T.block("pad_A"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                pad_A[v_i0, v_i1, v_i2, v_i3, v_i4] = A[v_i0, v_i1, v_i2, v_i3, v_i4]
        for nn, ff, yy, xx, zz, rc, ry, rx, rz in T.grid(14308, 1280, 1, 1, 1, 3, 2, 14, 14):
            with T.block("C"):
                v_nn, v_ff, v_yy, v_xx, v_zz, v_rc, v_ry, v_rx, v_rz = T.axis.remap("SSSSSRRRR", [nn, ff, yy, xx, zz, rc, ry, rx, rz])
                with T.init():
                    C[v_nn, v_ff, v_yy, v_xx, v_zz] = T.float16(0.0)
                C[v_nn, v_ff, v_yy, v_xx, v_zz] += pad_A[v_nn, v_rc, v_yy * 2 + v_ry, v_xx * 14 + v_rx, v_zz * 14 + v_rz]* W[v_ff, v_rc, v_ry, v_rx, v_rz]

    @T.prim_func
    def expected(A: T.Buffer((14308, 3, 2, 14, 14), "float16"), W: T.Buffer((1280, 3, 2, 14, 14), "float16"), C: T.Buffer((14308, 1280, 1, 1, 1), "float16")):
        T.func_attr({"tir.is_scheduled": 1})
        # with T.block("root"):
        C_reindex_pad_local = T.alloc_buffer((1, 14336, 1280), "float16", scope="local")
        pad_A_reindex_pad_shared = T.alloc_buffer((1, 14336, 1184), "float16", scope="shared")
        W_reindex_pad_shared = T.alloc_buffer((1, 1280, 1184), "float16", scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(20, thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(448, thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(1, thread="vthread.y"):
                    for ax1_1 in T.thread_binding(1, thread="vthread.x"):
                        for ax2_2 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(8, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(4, 2):
                                    for ax2_3_1_init in T.vectorized(2):
                                        with T.block("C_init"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial(14336, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3_init)
                                            v2 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0_init * 2 + ax2_3_1_init)
                                            C_reindex_pad_local[0, v1, v2] = T.float16(0.0)
                                for ax3_0 in range(74):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(2):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with T.block("pad_A_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(14336, ax1_0 * 32 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = T.axis.spatial(1184, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < 14308 and v2 < 1176, A[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(4):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with T.block("W_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = T.axis.spatial(1184, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        W_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v2 < 1176, W[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(16, 4, 2):
                                        for ax2_3_1 in T.vectorized(2):
                                            with T.block("C_update"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(14336, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3)
                                                v2 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0 * 2 + ax2_3_1)
                                                v3 = T.axis.reduce(1184, ax3_0 * 16 + ax3_1)
                                                C_reindex_pad_local[0, v1, v2] = C_reindex_pad_local[0, v1, v2] + pad_A_reindex_pad_shared[0, v1, v3] * W_reindex_pad_shared[0, v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(1, 4, 2):
                                    for ax2_1_1 in T.vectorized(2):
                                        with T.block("C_reindex_pad_local"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial(14336, ax1_0 * 32 + ax1_2 * 4 + ax1)
                                            v2 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_2 * 4 + ax2_0 * 2 + ax2_1_1)
                                            T.where(ax1_0 * 32 + ax1_2 * 4 + ax1 < 14308)
                                            C[v1, v2, 0, 0, 0] = C_reindex_pad_local[v0, v1, v2]
    # fmt: on


if __name__ == "__main__":
    tvm.testing.main()
