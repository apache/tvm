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
            with Target("nvidia/geforce-rtx-3090-ti"):
                return dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)

        return transform


class TestGEMV(BaseBeforeAfter):
    # fmt: off

    @T.prim_func
    def before(lv1637: T.Buffer((1, 32, 1, 128), "float16"), p_lv1638: T.handle, p_lv1614: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int32()
        lv1638 = T.match_buffer(p_lv1638, (1, 32, n, 128), "float16")
        lv1614 = T.match_buffer(p_lv1614, (1, 1, 1, n), "float16")
        var_compute_intermediate = T.match_buffer(p_output0, (1, 32, 1, n))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((1, 32, 1, n), "float16")
        var_T_divide_intermediate = T.alloc_buffer((1, 32, 1, n), "float16")
        var_T_maximum_intermediate = T.alloc_buffer((1, 32, 1, n), "float16")
        var_T_minimum_intermediate = T.alloc_buffer((1, 32, 1, n), "float16")
        for i0, i1, i2, i3, k in T.grid(1, 32, 1, n, 128):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(lv1637[v_i0, v_i1, v_i2, v_k], lv1638[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv1637[v_i0, v_i1, v_i2, v_k] * lv1638[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(1, 32, 1, n):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
        for ax0, ax1, ax2, ax3 in T.grid(1, 32, 1, n):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
        for ax0, ax1, ax2, ax3 in T.grid(1, 32, 1, n):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1614[v_ax0, 0, v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1614[v_ax0, 0, v_ax2, v_ax3])
        for i0, i1, i2, i3 in T.grid(1, 32, 1, n):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
                T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])

    @T.prim_func
    def expected(lv1637: T.Buffer((1, 32, 1, 128), "float16"), p_lv1638: T.handle, p_lv1614: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int32()
        lv1638 = T.match_buffer(p_lv1638, (1, 32, n, 128), "float16")
        lv1614 = T.match_buffer(p_lv1614, (1, 1, 1, n), "float16")
        var_compute_intermediate = T.match_buffer(p_output0, (1, 32, 1, n))
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((1, 32, 1, n), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((32, 1, 32, 1, n), "float16", scope="local")
        lv1637_shared = T.alloc_buffer((1, 32, 1, 128), "float16", scope="shared")
        lv1637_shared_local = T.alloc_buffer((1, 32, 1, 128), "float16", scope="local")
        for ax0_fused in T.thread_binding(32, thread="blockIdx.y"):
            for ax1_fused_0 in T.thread_binding(n, thread="blockIdx.x"):
                for ax1_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                    for ax2_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                        for u in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            for ax0_ax1_ax2_ax3_fused_0 in range(1):
                                for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_ax3_fused_3 in T.vectorized(4):
                                            with T.block("lv1637_shared"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(32, ax0_fused)
                                                v2 = T.axis.spatial(1, 0)
                                                v3 = T.axis.spatial(128, ax0_ax1_ax2_ax3_fused_0 * 128 + ax0_ax1_ax2_ax3_fused_1 * 128 + ax0_ax1_ax2_ax3_fused_2 * 4 + ax0_ax1_ax2_ax3_fused_3)
                                                T.reads(lv1637[v0, v1, v2, v3])
                                                T.writes(lv1637_shared[v0, v1, v2, v3])
                                                lv1637_shared[v0, v1, v2, v3] = lv1637[v0, v1, v2, v3]
                            with T.block("NT_matmul_rf_init"):
                                vax2_fused_1, v0 = T.axis.remap("SS", [ax2_fused_1, ax0_fused])
                                v1 = T.axis.spatial(n, ax1_fused_0 + ax1_fused_1)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1])
                                var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1] = T.float16(0)
                            for ax2_fused_0 in range(4):
                                for ax0_ax1_ax2_ax3_fused in T.vectorized(1):
                                    with T.block("lv1637_shared_local"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(32, ax0_fused)
                                        v2 = T.axis.spatial(1, 0)
                                        v3 = T.axis.spatial(128, ax2_fused_0 * 32 + ax2_fused_1)
                                        T.reads(lv1637_shared[v0, v1, v2, v3])
                                        T.writes(lv1637_shared_local[v0, v1, v2, v3])
                                        lv1637_shared_local[v0, v1, v2, v3] = lv1637_shared[v0, v1, v2, v3]
                                for u_1 in range(1):
                                    with T.block("NT_matmul_rf_update"):
                                        vax2_fused_1, v0 = T.axis.remap("SS", [ax2_fused_1, ax0_fused])
                                        v1 = T.axis.spatial(n, ax1_fused_0 + ax1_fused_1)
                                        vax2_fused_0 = T.axis.reduce(4, ax2_fused_0)
                                        T.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1], lv1637_shared_local[0, v0, 0, vax2_fused_0 * 32 + vax2_fused_1], lv1638[0, v0, v1, vax2_fused_0 * 32 + vax2_fused_1])
                                        T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1])
                                        var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1] = var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1] + lv1637_shared_local[0, v0, 0, vax2_fused_0 * 32 + vax2_fused_1] * lv1638[0, v0, v1, vax2_fused_0 * 32 + vax2_fused_1]
                    for ax1_ax2_fused in range(1):
                        for ax0 in T.thread_binding(32, thread="threadIdx.x"):
                            with T.block("NT_matmul"):
                                vax2_fused_1, v0, v1 = T.axis.remap("RSS", [ax0, ax0_fused, ax1_fused_0])
                                T.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1])
                                T.writes(var_NT_matmul_intermediate_local[0, v0, 0, v1])
                                with T.init():
                                    var_NT_matmul_intermediate_local[0, v0, 0, v1] = T.float16(0)
                                var_NT_matmul_intermediate_local[0, v0, 0, v1] = var_NT_matmul_intermediate_local[0, v0, 0, v1] + var_NT_matmul_intermediate_rf_local[vax2_fused_1, 0, v0, 0, v1]
                    with T.block("compute"):
                        v0, v1 = T.axis.remap("SS", [ax0_fused, ax1_fused_0])
                        T.reads(var_NT_matmul_intermediate_local[0, v0, 0, v1], lv1614[0, 0, 0, v1])
                        T.writes(var_compute_intermediate[0, v0, 0, v1])
                        var_compute_intermediate[0, v0, 0, v1] = T.Cast("float32", T.min(T.max(var_NT_matmul_intermediate_local[0, v0, 0, v1] * T.float16(0.088397790055248615), T.float16(-65504)), lv1614[0, 0, 0, v1]))

    # fmt: on


class TestDecodeGEMV1(BaseBeforeAfter):
    # fmt: off

    @T.prim_func
    def before(lv571: T.Buffer((22016, 512), "uint32"), lv572: T.Buffer((22016, 128), "float16"), lv1654: T.Buffer((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Buffer((1, 1, 22016), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        p_output0_intermediate = T.alloc_buffer((22016, 4096), "float16")
        for i, j in T.grid(22016, 4096):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv571[v_i, v_j // 8], lv572[v_i, v_j // 32])
                T.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v_i, v_j // 32]
        for i0, i1, i2, k in T.grid(1, 1, 22016, 4096):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @T.prim_func
    def expected(lv571: T.Buffer((22016, 512), "uint32"), lv572: T.Buffer((22016, 128), "float16"), lv1654: T.Buffer((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Buffer((1, 1, 22016), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((32, 1, 1, 22016), "float16", scope="local")
        lv1654_shared = T.alloc_buffer((1, 1, 4096), "float16", scope="shared")
        lv1654_shared_local = T.alloc_buffer((1, 1, 4096), "float16", scope="local")
        for u_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax0_fused_0 in T.thread_binding(2752, thread="blockIdx.x"):
                for ax0_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                    for ax1_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                        for u in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            for ax0_ax1_ax2_fused_0 in range(2):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_fused_3 in T.vectorized(8):
                                            with T.block("lv1654_shared"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(1, 0)
                                                v2 = T.axis.spatial(4096, ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 256 + ax0_ax1_ax2_fused_2 * 8 + ax0_ax1_ax2_fused_3)
                                                T.reads(lv1654[v0, v1, v2])
                                                T.writes(lv1654_shared[v0, v1, v2])
                                                lv1654_shared[v0, v1, v2] = lv1654[v0, v1, v2]
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_1 = T.axis.spatial(32, ax1_0_fused_1)
                                v0 = T.axis.spatial(22016, ax0_fused_0 * 8 + ax0_fused_1)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                            for ax1_0_fused_0 in range(16):
                                for ax0_ax1_ax2_fused_0 in range(1):
                                    for ax0_ax1_ax2_fused_1 in T.vectorized(8):
                                        with T.block("lv1654_shared_local"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial(1, 0)
                                            v2 = T.axis.spatial(4096, ax1_0_fused_0 * 256 + ax1_0_fused_1 * 8 + ax0_ax1_ax2_fused_0 * 8 + ax0_ax1_ax2_fused_1)
                                            T.reads(lv1654_shared[v0, v1, v2])
                                            T.writes(lv1654_shared_local[v0, v1, v2])
                                            lv1654_shared_local[v0, v1, v2] = lv1654_shared[v0, v1, v2]
                                for ax1_1 in range(8):
                                    with T.block("NT_matmul_rf_update"):
                                        vax1_0_fused_1 = T.axis.spatial(32, ax1_0_fused_1)
                                        v0 = T.axis.spatial(22016, ax0_fused_0 * 8 + ax0_fused_1)
                                        vax1_0_fused_0, vax1_1 = T.axis.remap("RR", [ax1_0_fused_0, ax1_1])
                                        T.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0], lv1654_shared_local[0, 0, vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1], lv571[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 8], lv572[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                        T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                        var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] + lv1654_shared_local[0, 0, vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.thread_binding(32, thread="threadIdx.x"):
                            with T.block("NT_matmul"):
                                vax1_0_fused_1 = T.axis.reduce(32, ax0)
                                v0 = T.axis.spatial(22016, ax0_fused_0 * 8 + ax0_fused_1)
                                T.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                T.writes(var_NT_matmul_intermediate[0, 0, v0])
                                with T.init():
                                    var_NT_matmul_intermediate[0, 0, v0] = T.float16(0)
                                var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0]

    # fmt: on


class TestDecodeGEMV2(BaseBeforeAfter):
    # fmt: off

    @T.prim_func
    def before(lv771: T.Buffer((32000, 512), "uint32"), lv772: T.Buffer((32000, 128), "float16"), lv3216: T.Buffer((1, 1, 4096), "float16"), p_output0_intermediate: T.Buffer((1, 1, 32000), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((32000, 4096), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((1, 1, 32000), "float16")
        for i, j in T.grid(32000, 4096):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv771[v_i, v_j // 8], lv772[v_i, v_j // 32])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv771[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv772[v_i, v_j // 32]
        for i0, i1, i2, k in T.grid(1, 1, 32000, 4096):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv3216[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv3216[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for i0, i1, i2 in T.grid(1, 1, 32000):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def expected(lv771: T.Buffer((32000, 512), "uint32"), lv772: T.Buffer((32000, 128), "float16"), lv3216: T.Buffer((1, 1, 4096), "float16"), p_output0_intermediate: T.Buffer((1, 1, 32000), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((1, 1, 32000), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((32, 1, 1, 32000), "float16", scope="local")
        lv3216_shared = T.alloc_buffer((1, 1, 4096), "float16", scope="shared")
        lv3216_shared_local = T.alloc_buffer((1, 1, 4096), "float16", scope="local")
        for u_fused in T.thread_binding(1, thread="blockIdx.y"):
            for ax0_fused_0 in T.thread_binding(4000, thread="blockIdx.x"):
                for ax0_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                    for ax1_0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                        for u in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            for ax0_ax1_ax2_fused_0 in range(2):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_fused_3 in T.vectorized(8):
                                            with T.block("lv3216_shared"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(1, 0)
                                                v2 = T.axis.spatial(4096, ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 256 + ax0_ax1_ax2_fused_2 * 8 + ax0_ax1_ax2_fused_3)
                                                T.reads(lv3216[v0, v1, v2])
                                                T.writes(lv3216_shared[v0, v1, v2])
                                                lv3216_shared[v0, v1, v2] = lv3216[v0, v1, v2]
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_1 = T.axis.spatial(32, ax1_0_fused_1)
                                v0 = T.axis.spatial(32000, ax0_fused_0 * 8 + ax0_fused_1)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                            for ax1_0_fused_0 in range(16):
                                for ax0_ax1_ax2_fused_0 in range(1):
                                    for ax0_ax1_ax2_fused_1 in T.vectorized(8):
                                        with T.block("lv3216_shared_local"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial(1, 0)
                                            v2 = T.axis.spatial(4096, ax1_0_fused_0 * 256 + ax1_0_fused_1 * 8 + ax0_ax1_ax2_fused_0 * 8 + ax0_ax1_ax2_fused_1)
                                            T.reads(lv3216_shared[v0, v1, v2])
                                            T.writes(lv3216_shared_local[v0, v1, v2])
                                            lv3216_shared_local[v0, v1, v2] = lv3216_shared[v0, v1, v2]
                                for ax1_1 in range(8):
                                    with T.block("NT_matmul_rf_update"):
                                        vax1_0_fused_1 = T.axis.spatial(32, ax1_0_fused_1)
                                        v0 = T.axis.spatial(32000, ax0_fused_0 * 8 + ax0_fused_1)
                                        vax1_0_fused_0, vax1_1 = T.axis.remap("RR", [ax1_0_fused_0, ax1_1])
                                        T.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0], lv3216_shared_local[0, 0, vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1], lv771[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 8], lv772[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                        T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                        var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] + lv3216_shared_local[0, 0, vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv771[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv772[v0, (vax1_0_fused_0 * 256 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.thread_binding(32, thread="threadIdx.x"):
                            with T.block("NT_matmul"):
                                vax1_0_fused_1 = T.axis.reduce(32, ax0)
                                v0 = T.axis.spatial(32000, ax0_fused_0 * 8 + ax0_fused_1)
                                T.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                T.writes(var_NT_matmul_intermediate_local[0, 0, v0])
                                with T.init():
                                    var_NT_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                                var_NT_matmul_intermediate_local[0, 0, v0] = var_NT_matmul_intermediate_local[0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0]
                    with T.block("compute"):
                        v0 = T.axis.spatial(32000, ax0_fused_0 * 8 + ax0_fused_1)
                        T.reads(var_NT_matmul_intermediate_local[0, 0, v0])
                        T.writes(p_output0_intermediate[0, 0, v0])
                        p_output0_intermediate[0, 0, v0] = T.Cast("float32", var_NT_matmul_intermediate_local[0, 0, v0])

    # fmt: on


if __name__ == "__main__":
    tvm.testing.main()
