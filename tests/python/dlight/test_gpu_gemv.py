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
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((128, 1, 32, 1, n), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((64, 1, 32, 1, n), "float16", scope="local")
        lv1638_local = T.alloc_buffer((1, 32, n, 128), "float16", scope="local")
        lv1637_shared = T.alloc_buffer((1, 32, 1, 128), "float16", scope="shared")
        for ax0_fused_ax1_fused_fused_0 in T.thread_binding(n * 32, thread="blockIdx.x"):
            for ax0_fused_ax1_fused_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax0, ax1, ax2 in T.grid(1, 1, 1):
                        for ax3_0 in T.serial(1, annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                            for ax3_1 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax3_2 in T.thread_binding(64, thread="threadIdx.x"):
                                    for ax3_3 in T.vectorized(2):
                                        with T.block("lv1637_shared"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n + ax1)
                                            v2 = T.axis.spatial(1, ax2)
                                            v3 = T.axis.spatial(128, ax3_0 * 128 + ax3_1 * 128 + ax3_2 * 2 + ax3_3)
                                            T.reads(lv1637[v0, v1, v2, v3])
                                            T.writes(lv1637_shared[v0, v1, v2, v3])
                                            lv1637_shared[v0, v1, v2, v3] = lv1637[v0, v1, v2, v3]
                    for ax0_fused_ax1_fused_fused_2_init in range(1):
                        for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(2):
                            with T.block("NT_matmul_rf_init"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(128, ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                v0 = T.axis.spatial(32, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2_init) // n)
                                v1 = T.axis.spatial(n, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2_init) % n)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1])
                                var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1] = T.float16(0)
                    for ax2_fused_u_fused_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0, ax1, ax2_0, ax3 in T.grid(1, 1, 1, 2):
                            for ax2_1 in T.vectorized(1):
                                with T.block("lv1638_local"):
                                    v0 = T.axis.spatial(1, ax0)
                                    v1 = T.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n + ax1)
                                    v2 = T.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n + ax2_0 + ax2_1)
                                    v3 = T.axis.spatial(128, ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + ax3)
                                    T.reads(lv1638[v0, v1, v2, v3])
                                    T.writes(lv1638_local[v0, v1, v2, v3])
                                    lv1638_local[v0, v1, v2, v3] = lv1638[v0, v1, v2, v3]
                        for ax0_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(1, 1):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(2):
                                with T.block("NT_matmul_rf_update"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(128, ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                    v0 = T.axis.spatial(32, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2) // n)
                                    v1 = T.axis.spatial(n, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2) % n)
                                    vax2_fused_u_fused_2, vax2_fused_u_fused_0 = T.axis.remap("RR", [ax2_fused_u_fused_2, ax2_fused_u_fused_0])
                                    T.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1], lv1637_shared[0, v0, 0, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused], lv1638_local[0, v0, v1, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused])
                                    T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1])
                                    var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1] = var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1] + lv1637_shared[0, v0, 0, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused] * lv1638_local[0, v0, v1, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused]
            for ax2_ax3_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                for ax0 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax2_ax3_fused_1_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax2_ax3_fused_1_1 in T.vectorized(1):
                            with T.block("NT_matmul_rf_init"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.spatial(64, ax0)
                                v0 = T.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                                v1 = T.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1])
                                var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1] = T.float16(0)
                            for ax1 in range(2):
                                with T.block("NT_matmul_rf_update"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                    v0 = T.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                                    v1 = T.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                                    T.reads(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1], var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, 0, v0, 0, v1])
                                    T.writes(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1])
                                    var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1] = var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1] + var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, 0, v0, 0, v1]
            for ax1_ax2_fused_1 in range(1):
                for ax1_ax2_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                    for ax0 in T.thread_binding(64, thread="threadIdx.x"):
                        with T.block("NT_matmul"):
                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.reduce(64, ax0)
                            v0 = T.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                            v1 = T.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                            T.reads(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1])
                            T.writes(var_NT_matmul_intermediate_local[0, v0, 0, v1])
                            with T.init():
                                var_NT_matmul_intermediate_local[0, v0, 0, v1] = T.float16(0)
                            var_NT_matmul_intermediate_local[0, v0, 0, v1] = var_NT_matmul_intermediate_local[0, v0, 0, v1] + var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1]
            for ax0_ax1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                for ax0_ax1_fused_1 in range(1):
                    with T.block("compute"):
                        v0 = T.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                        v1 = T.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                        T.reads(var_NT_matmul_intermediate_local[0, v0, 0, v1], lv1614[0, 0, 0, v1])
                        T.writes(var_compute_intermediate[0, v0, 0, v1])
                        var_compute_intermediate[0, v0, 0, v1] = T.Cast("float32", T.min(T.max(var_NT_matmul_intermediate_local[0, v0, 0, v1] * T.float16(0.088397790055248615), T.float16(-65504)), lv1614[0, 0, 0, v1]))

    # fmt: on


def test_decode_gemv1():
    # fmt: off

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def expected(lv571: T.Buffer((22016, 512), "uint32"), lv572: T.Buffer((22016, 128), "float16"), lv1654: T.Buffer((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Buffer((1, 1, 22016), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((256, 1, 1, 22016), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((64, 1, 1, 22016), "float16", scope="local")
        lv571_local = T.alloc_buffer((22016, 512), "uint32", scope="local")
        lv1654_shared = T.alloc_buffer((1, 1, 4096), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.thread_binding(5504, thread="blockIdx.x"):
            for u_fused_ax0_fused_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax0, ax1 in T.grid(1, 1):
                        for ax2_0 in T.serial(2, annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                            for ax2_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax2_2 in T.thread_binding(64, thread="threadIdx.x"):
                                    for ax2_3 in T.vectorized(8):
                                        with T.block("lv1654_shared"):
                                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                            v2 = T.axis.spatial(4096, ax2_0 * 2048 + ax2_1 * 512 + ax2_2 * 8 + ax2_3)
                                            T.reads(lv1654[v0, v1, v2])
                                            T.writes(lv1654_shared[v0, v1, v2])
                                            lv1654_shared[v0, v1, v2] = lv1654[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in range(1):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(4):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(256, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = T.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = T.float16(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(8, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0_0, ax1 in T.grid(1, 1):
                            for ax0_1 in T.vectorized(1):
                                with T.block("lv571_local"):
                                    v0 = T.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + ax0_0 + ax0_1)
                                    v1 = T.axis.spatial(512, ax1_0_fused_ax1_1_fused_0 * 64 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1)
                                    T.reads(lv571[v0, v1])
                                    T.writes(lv571_local[v0, v1])
                                    lv571_local[v0, v1] = lv571[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(1, 2):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(4):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(256, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = T.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = T.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    T.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0], lv1654_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4], lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 64 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
                                    T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] + lv1654_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 64 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
            for ax2_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                for ax0 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax2_fused_1_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax2_fused_1_1 in T.vectorized(1):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.spatial(64, ax0)
                                v0 = T.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = T.float16(0)
                            for ax1 in range(4):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                    v0 = T.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                    T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0])
                                    T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0]
            for ax1_fused_1 in range(1):
                for ax1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for ax0 in T.thread_binding(64, thread="threadIdx.x"):
                        with T.block("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.reduce(64, ax0)
                            v0 = T.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + ax1_fused_0 + ax1_fused_1)
                            T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                            T.writes(var_NT_matmul_intermediate[0, 0, v0])
                            with T.init():
                                var_NT_matmul_intermediate[0, 0, v0] = T.float16(0)
                            var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0]

    # fmt: on

    mod = tvm.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_decode_gemv2():
    # fmt: off

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def expected(lv771: T.Buffer((32000, 512), "uint32"), lv772: T.Buffer((32000, 128), "float16"), lv3216: T.Buffer((1, 1, 4096), "float16"), p_output0_intermediate: T.Buffer((1, 1, 32000), "float32")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((1, 1, 32000), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((256, 1, 1, 32000), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((64, 1, 1, 32000), "float16", scope="local")
        lv771_local = T.alloc_buffer((32000, 512), "uint32", scope="local")
        lv3216_shared = T.alloc_buffer((1, 1, 4096), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.thread_binding(8000, thread="blockIdx.x"):
            for u_fused_ax0_fused_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax0, ax1 in T.grid(1, 1):
                        for ax2_0 in T.serial(2, annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                            for ax2_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax2_2 in T.thread_binding(64, thread="threadIdx.x"):
                                    for ax2_3 in T.vectorized(8):
                                        with T.block("lv3216_shared"):
                                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                            v2 = T.axis.spatial(4096, ax2_0 * 2048 + ax2_1 * 512 + ax2_2 * 8 + ax2_3)
                                            T.reads(lv3216[v0, v1, v2])
                                            T.writes(lv3216_shared[v0, v1, v2])
                                            lv3216_shared[v0, v1, v2] = lv3216[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in range(1):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(4):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(256, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = T.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = T.float16(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(8, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0_0, ax1 in T.grid(1, 1):
                            for ax0_1 in T.vectorized(1):
                                with T.block("lv771_local"):
                                    v0 = T.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + ax0_0 + ax0_1)
                                    v1 = T.axis.spatial(512, ax1_0_fused_ax1_1_fused_0 * 64 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1)
                                    T.reads(lv771[v0, v1])
                                    T.writes(lv771_local[v0, v1])
                                    lv771_local[v0, v1] = lv771[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(1, 2):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(4):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(256, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = T.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = T.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    T.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0], lv3216_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4], lv771_local[v0, vax1_0_fused_ax1_1_fused_0 * 64 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], lv772[v0, (vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
                                    T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] + lv3216_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv771_local[v0, vax1_0_fused_ax1_1_fused_0 * 64 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv772[v0, (vax1_0_fused_ax1_1_fused_0 * 512 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
            for ax2_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                for ax0 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax2_fused_1_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax2_fused_1_1 in T.vectorized(1):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.spatial(64, ax0)
                                v0 = T.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = T.float16(0)
                            for ax1 in range(4):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                    v0 = T.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                    T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0])
                                    T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0]
            for ax1_fused_1 in range(1):
                for ax1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                    for ax0 in T.thread_binding(64, thread="threadIdx.x"):
                        with T.block("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.reduce(64, ax0)
                            v0 = T.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 4 + ax1_fused_0 + ax1_fused_1)
                            T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                            T.writes(var_NT_matmul_intermediate_local[0, 0, v0])
                            with T.init():
                                var_NT_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                            var_NT_matmul_intermediate_local[0, 0, v0] = var_NT_matmul_intermediate_local[0, 0, v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0]
            for ax0_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                for ax0_fused_1 in range(1):
                    with T.block("compute"):
                        v0 = T.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 4 + ax0_fused_0 + ax0_fused_1)
                        T.reads(var_NT_matmul_intermediate_local[0, 0, v0])
                        T.writes(p_output0_intermediate[0, 0, v0])
                        p_output0_intermediate[0, 0, v0] = T.Cast("float32", var_NT_matmul_intermediate_local[0, 0, v0])

    # fmt: on

    mod = tvm.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_decode_gemv3():
    # fmt: off

    @T.prim_func(private=True)
    def before(lv575: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv576: T.Buffer((T.int64(4096), T.int64(344)), "float16"), lv574: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv570: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(11008)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv575[v_i, v_j // T.int64(8)], lv576[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv574[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv574[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv570[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv570[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def expected(lv575: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv576: T.Buffer((T.int64(4096), T.int64(344)), "float16"), lv574: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv570: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(128), T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((T.int64(32), T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
        lv575_local = T.alloc_buffer((T.int64(4096), T.int64(1376)), "uint32", scope="local")
        lv574_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for u_fused_ax0_fused_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                        for ax2_0 in T.serial(T.int64(3), annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                            for ax2_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                for ax2_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax2_3 in T.vectorized(T.int64(8)):
                                        with T.block("lv574_shared"):
                                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                            v2 = T.axis.spatial(T.int64(11008), ax2_0 * T.int64(4096) + ax2_1 * T.int64(256) + ax2_2 * T.int64(8) + ax2_3)
                                            T.where(((ax2_0 * T.int64(16) + ax2_1) * T.int64(32) + ax2_2) * T.int64(8) + ax2_3 < T.int64(11008))
                                            T.reads(lv574[v0, v1, v2])
                                            T.writes(lv574_shared[v0, v1, v2])
                                            lv574_shared[v0, v1, v2] = lv574[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in range(T.int64(1)):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(T.int64(128), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(T.int64(43), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0_0, ax1 in T.grid(T.int64(1), T.int64(1)):
                            for ax0_1 in T.vectorized(T.int64(1)):
                                with T.block("lv575_local"):
                                    v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + ax0_0 + ax0_1)
                                    v1 = T.axis.spatial(T.int64(1376), ax1_0_fused_ax1_1_fused_0 * T.int64(32) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1)
                                    T.reads(lv575[v0, v1])
                                    T.writes(lv575_local[v0, v1])
                                    lv575_local[v0, v1] = lv575[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(T.int64(1), T.int64(2)):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(T.int64(128), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = T.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    T.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0], lv574_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)], lv575_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], lv576[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
                                    T.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] + lv574_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv575_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
            for ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for ax2_fused_1_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax2_fused_1_1 in T.vectorized(T.int64(1)):
                            with T.block("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.spatial(T.int64(32), ax0)
                                v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                T.reads()
                                T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = T.float16(0)
                            for ax1 in range(T.int64(4)):
                                with T.block("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                    v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                    T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0])
                                    T.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0]
            for ax1_fused_1 in range(T.int64(1)):
                for ax1_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        with T.block("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.reduce(T.int64(32), ax0)
                            v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax1_fused_0 + ax1_fused_1)
                            T.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                            T.writes(var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                            with T.init():
                                var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                            var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0]
            for ax0_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0_fused_1 in range(T.int64(1)):
                    with T.block("T_add"):
                        v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax0_fused_0 + ax0_fused_1)
                        T.reads(lv570[T.int64(0), T.int64(0), v0], var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        T.writes(p_output0_intermediate[T.int64(0), T.int64(0), v0])
                        p_output0_intermediate[T.int64(0), T.int64(0), v0] = lv570[T.int64(0), T.int64(0), v0] + var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0]

    # fmt: on

    mod = tvm.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
