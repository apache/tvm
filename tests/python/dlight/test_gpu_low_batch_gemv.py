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

import tvm.testing
from tvm import dlight as dl
from tvm.script import tir as T
from tvm.target import Target


def test_batch_decode_gemv():
    # fmt: off

    @T.prim_func(private=True)
    def before(lv429: T.Buffer((T.int64(4096), T.int64(3584)), "uint32"), lv430: T.Buffer((T.int64(4096), T.int64(896)), "float16"), p_lv807: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True), "tir.HoistIfThenElseExprWithBlock": 1})
        batch_size = T.int64()
        lv807 = T.match_buffer(p_lv807, (batch_size, T.int64(1), T.int64(28672)), "float16")
        NT_matmul_intermediate = T.match_buffer(p_output0, (batch_size, T.int64(1), T.int64(4096)), "float16")
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(4096), T.int64(28672)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(28672)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(28672)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv429[v_i0, v_i1 // T.int64(8)])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv429[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(28672)):
            with T.block("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], lv430[v_i0, v_i1 // T.int64(32)])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * lv430[v_i0, v_i1 // T.int64(32)]
        for i0, i1, i2, k in T.grid(batch_size, T.int64(1), T.int64(4096), T.int64(28672)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv807[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv807[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_i2, v_k]

    @T.prim_func(private=True)
    def expected(lv429: T.Buffer((T.int64(4096), T.int64(3584)), "uint32"), lv430: T.Buffer((T.int64(4096), T.int64(896)), "float16"), p_lv807: T.handle, p_output0: T.handle):
        T.func_attr({"tir.HoistIfThenElseExprWithBlock": 1, "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        batch_size = T.int64()
        lv807 = T.match_buffer(p_lv807, (batch_size, T.int64(1), T.int64(28672)), "float16")
        NT_matmul_intermediate = T.match_buffer(p_output0, (batch_size, T.int64(1), T.int64(4096)), "float16")
        # with T.block("root"):
        dequantize_intermediate_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(28672)), "float16", scope="local")
        NT_matmul_intermediate_pad_local = T.alloc_buffer(((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_intermediate_pad_rf_local = T.alloc_buffer((T.int64(128), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_intermediate_pad_rf_local_1 = T.alloc_buffer((T.int64(32), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        for ax0_0 in T.thread_binding((batch_size + T.int64(3)) // T.int64(4), thread="blockIdx.y"):
            for u_fused_ax1_fused_fused_0 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                for u_fused_ax1_fused_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.thread_binding(T.int64(32), thread="threadIdx.y"):
                        for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(4), T.int64(2)):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                with T.block("NT_matmul_rf_init"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                    v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1_init)
                                    v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                    T.reads()
                                    T.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                    NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_u_fused_0 in T.serial(T.int64(112), annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                            for ax0_0_1, ax1 in T.grid(T.int64(2), T.int64(8)):
                                for ax0_1 in T.vectorized(T.int64(1)):
                                    with T.block("dequantize"):
                                        v0 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + ax0_0_1 + ax0_1)
                                        v1 = T.axis.spatial(T.int64(28672), ax2_fused_u_fused_0 * T.int64(256) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(8) + ax1)
                                        T.reads(lv429[v0, v1 // T.int64(8)], lv430[v0, v1 // T.int64(32)])
                                        T.writes(dequantize_intermediate_intermediate_local[v0, v1])
                                        dequantize_intermediate_intermediate_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv429[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7)) * lv430[v0, v1 // T.int64(32)]
                            for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(4), T.int64(2), T.int64(2)):
                                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                    with T.block("NT_matmul_rf_update"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                        v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1)
                                        v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                        vax2_fused_u_fused_0, vax2_fused_u_fused_2 = T.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                        T.reads(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1], lv807[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                        T.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                        NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] + T.if_then_else(v0 < batch_size, lv807[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                for ax3_fused_0_ax3_fused_1_fused in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.y"):
                        for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                            for ax2 in range(T.int64(4)):
                                for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                    with T.block("NT_matmul_rf_init"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.spatial(T.int64(32), ax0)
                                        v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                        v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                        T.reads()
                                        T.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                        NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = T.float16(0)
                                    for ax1 in range(T.int64(4)):
                                        with T.block("NT_matmul_rf_update"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                            v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                            v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                            T.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1], NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1])
                                            T.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                            NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] + NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1]
                for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(4)):
                    for ax2_fused_0_ax2_fused_1_fused in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                        for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.y"):
                            with T.block("NT_matmul"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.reduce(T.int64(32), ax0)
                                v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax1)
                                v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                T.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                T.writes(NT_matmul_intermediate_pad_local[v0, T.int64(0), v1])
                                with T.init():
                                    NT_matmul_intermediate_pad_local[v0, T.int64(0), v1] = T.float16(0)
                                NT_matmul_intermediate_pad_local[v0, T.int64(0), v1] = NT_matmul_intermediate_pad_local[v0, T.int64(0), v1] + NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1]
                for ax0 in range(T.int64(4)):
                    for ax1_fused_0_ax1_fused_1_fused in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                        for ax1_fused_2 in range(T.int64(2)):
                            with T.block("NT_matmul_intermediate_pad"):
                                v0 = T.axis.spatial(batch_size, ax0_0 * T.int64(4) + ax0)
                                v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                T.where((ax0_0 - (batch_size + T.int64(3)) // T.int64(4) < T.int64(0) or ax0_0 == T.int64(0)) and ax0_0 * T.int64(4) + ax0 < batch_size)
                                T.reads(NT_matmul_intermediate_pad_local[v0, T.int64(0), v1])
                                T.writes(NT_matmul_intermediate[v0, T.int64(0), v1])
                                NT_matmul_intermediate[v0, T.int64(0), v1] = NT_matmul_intermediate_pad_local[v0, T.int64(0), v1]

    # fmt: on
    mod = tvm.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4))(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_batch_gemv():
    N = 4096
    K = 4096
    # fmt: off
    @T.prim_func(private=True)
    def before(var_A: T.handle, B: T.Buffer((T.int64(N), T.int64(K)), "float16"), var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True), "tir.HoistIfThenElseExprWithBlock": 1})
        batch_size = T.int64()
        A = T.match_buffer(var_A, (batch_size, T.int64(1), T.int64(K)), "float16")
        NT_matmul = T.match_buffer(var_NT_matmul, (batch_size, T.int64(1), T.int64(N)), "float16")
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(batch_size, T.int64(1), T.int64(N), T.int64(K)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def expected(var_A: T.handle, B: T.Buffer((T.int64(4096), T.int64(4096)), "float16"), var_NT_matmul: T.handle):
        T.func_attr({"tir.HoistIfThenElseExprWithBlock": 1, "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        batch_size = T.int64()
        A = T.match_buffer(var_A, (batch_size, T.int64(1), T.int64(4096)), "float16")
        NT_matmul = T.match_buffer(var_NT_matmul, (batch_size, T.int64(1), T.int64(4096)), "float16")
        # with T.block("root"):
        NT_matmul_pad_local = T.alloc_buffer(((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_pad_rf_local = T.alloc_buffer((T.int64(128), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_pad_rf_local_1 = T.alloc_buffer((T.int64(32), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        for ax0_0 in T.thread_binding((batch_size + T.int64(3)) // T.int64(4), thread="blockIdx.y"):
            for u_fused_ax1_fused_fused_0 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                for u_fused_ax1_fused_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.thread_binding(T.int64(32), thread="threadIdx.y"):
                        for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(4), T.int64(2)):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                with T.block("NT_matmul_rf_init"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                    v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1_init)
                                    v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                    T.reads()
                                    T.writes(NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                    NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_u_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                            for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(4), T.int64(2), T.int64(2)):
                                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                    with T.block("NT_matmul_rf_update"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                        v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1)
                                        v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                        vax2_fused_u_fused_0, vax2_fused_u_fused_2 = T.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                        T.reads(NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1], A[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                        T.writes(NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                        NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] + T.if_then_else(v0 < batch_size, A[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                for ax3_fused_0_ax3_fused_1_fused in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.y"):
                        for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 8, "pragma_unroll_explicit": 1}):
                            for ax2 in range(T.int64(4)):
                                for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                    with T.block("NT_matmul_rf_init"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.spatial(T.int64(32), ax0)
                                        v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                        v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                        T.reads()
                                        T.writes(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                        NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = T.float16(0)
                                    for ax1 in range(T.int64(4)):
                                        with T.block("NT_matmul_rf_update"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                            v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                            v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                            T.reads(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1], NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1])
                                            T.writes(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                            NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] + NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1]
                for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(4)):
                    for ax2_fused_0_ax2_fused_1_fused in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                        for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.y"):
                            with T.block("NT_matmul"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.reduce(T.int64(32), ax0)
                                v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax1)
                                v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                T.reads(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                T.writes(NT_matmul_pad_local[v0, T.int64(0), v1])
                                with T.init():
                                    NT_matmul_pad_local[v0, T.int64(0), v1] = T.float16(0)
                                NT_matmul_pad_local[v0, T.int64(0), v1] = NT_matmul_pad_local[v0, T.int64(0), v1] + NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1]
                for ax0 in range(T.int64(4)):
                    for ax1_fused_0_ax1_fused_1_fused in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                        for ax1_fused_2 in range(T.int64(2)):
                            with T.block("NT_matmul_pad"):
                                v0 = T.axis.spatial(batch_size, ax0_0 * T.int64(4) + ax0)
                                v1 = T.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                T.where((ax0_0 - (batch_size + T.int64(3)) // T.int64(4) < T.int64(0) or ax0_0 == T.int64(0)) and ax0_0 * T.int64(4) + ax0 < batch_size)
                                T.reads(NT_matmul_pad_local[v0, T.int64(0), v1])
                                T.writes(NT_matmul[v0, T.int64(0), v1])
                                NT_matmul[v0, T.int64(0), v1] = NT_matmul_pad_local[v0, T.int64(0), v1]
    # fmt: on
    mod = tvm.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4))(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_reduction_symbolic_var():
    # fmt: off
    @T.prim_func(private=True)
    def before(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        kv_seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), kv_seq_len))
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), kv_seq_len, T.int64(128)))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), kv_seq_len):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
    # fmt: on
    mod = tvm.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4))(mod)
    tvm.ir.assert_structural_equal(mod["main"], before)


def test_small_spatial_axis():
    @T.prim_func(private=True)
    def func(var_A: T.handle, B: T.Buffer((T.int64(8), T.int64(4096)), "float16"), var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        batch_size = T.int64()
        A = T.match_buffer(var_A, (batch_size, T.int64(4096)), "float16")
        C = T.match_buffer(var_C, (batch_size, T.int64(8)), "float16")
        for i0, i1, k in T.grid(batch_size, T.int64(8), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_k], B[v_i1, v_k])
                T.writes(C[v_i0, v_i1])
                with T.init():
                    C[v_i0, v_i1] = T.float16(0)
                C[v_i0, v_i1] = C[v_i0, v_i1] + A[v_i0, v_k] * B[v_i1, v_k]

    # fmt: off
    @T.prim_func(private=True)
    def expected(var_A: T.handle, B: T.Buffer((T.int64(8), T.int64(4096)), "float16"), var_C: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        batch_size = T.int64()
        A = T.match_buffer(var_A, (batch_size, T.int64(4096)), "float16")
        C = T.match_buffer(var_C, (batch_size, T.int64(8)), "float16")
        # with T.block("root"):
        C_pad_local = T.alloc_buffer(((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(8)), "float16", scope="local")
        C_pad_rf_local = T.alloc_buffer((T.int64(128), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(8)), "float16", scope="local")
        C_pad_rf_local_1 = T.alloc_buffer((T.int64(32), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(8)), "float16", scope="local")
        for ax0_0 in T.thread_binding((batch_size + T.int64(3)) // T.int64(4), thread="blockIdx.y"):
            for u_fused_ax1_fused_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for u_fused_ax1_fused_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(4), T.int64(2)):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                with T.block("NT_matmul_rf_init"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                    v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1_init)
                                    v1 = T.axis.spatial(T.int64(8), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                    T.where((u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1) * T.int64(2) + u_fused_ax1_fused_fused_2_init < T.int64(8))
                                    T.reads()
                                    T.writes(C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1])
                                    C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1] = T.float16(0)
                        for ax2_fused_u_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(4), T.int64(2), T.int64(2)):
                                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                    with T.block("NT_matmul_rf_update"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                        v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1)
                                        v1 = T.axis.spatial(T.int64(8), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                        vax2_fused_u_fused_0, vax2_fused_u_fused_2 = T.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                        T.where((u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1) * T.int64(2) + u_fused_ax1_fused_fused_2 < T.int64(8))
                                        T.reads(C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1], A[v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                        T.writes(C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1])
                                        C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1] = C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1] + T.if_then_else(v0 < batch_size, A[v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                for ax3_fused_0_ax3_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                    for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                        for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            for ax2 in range(T.int64(4)):
                                for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                    with T.block("NT_matmul_rf_init"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.spatial(T.int64(32), ax0)
                                        v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                        v1 = T.axis.spatial(T.int64(8), ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                        T.where((T.Mul(T.int64(0), T.int64(16)) + ax3_fused_0_ax3_fused_1_fused % T.int64(16)) * T.int64(2) + (ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1) < T.int64(8))
                                        T.reads()
                                        T.writes(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1])
                                        C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1] = T.float16(0)
                                    for ax1 in range(T.int64(4)):
                                        with T.block("NT_matmul_rf_update"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                            v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                            v1 = T.axis.spatial(T.int64(8), ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                            T.where((T.Mul(T.int64(0), T.int64(16)) + ax3_fused_0_ax3_fused_1_fused % T.int64(16)) * T.int64(2) + (ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1) < T.int64(8))
                                            T.reads(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1], C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, v1])
                                            T.writes(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1])
                                            C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1] = C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1] + C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, v1]
                for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(4)):
                    for ax2_fused_0_ax2_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            with T.block("NT_matmul"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.reduce(T.int64(32), ax0)
                                v0 = T.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax1)
                                v1 = T.axis.spatial(T.int64(8), ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                T.where((T.Mul(T.int64(0), T.int64(16)) + ax2_fused_0_ax2_fused_1_fused % T.int64(16)) * T.int64(2) + ax2_fused_2 < T.int64(8))
                                T.reads(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1])
                                T.writes(C_pad_local[v0, v1])
                                with T.init():
                                    C_pad_local[v0, v1] = T.float16(0)
                                C_pad_local[v0, v1] = C_pad_local[v0, v1] + C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1]
                for ax0 in range(T.int64(4)):
                    for ax1_fused_0_ax1_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax1_fused_2 in range(T.int64(2)):
                            with T.block("C_pad"):
                                v0 = T.axis.spatial(batch_size, ax0_0 * T.int64(4) + ax0)
                                v1 = T.axis.spatial(T.int64(8), ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                T.where((ax0_0 - (batch_size + T.int64(3)) // T.int64(4) < T.int64(0) or ax0_0 == T.int64(0)) and ax0_0 * T.int64(4) + ax0 < batch_size and (T.Mul(T.int64(0), T.int64(16)) + ax1_fused_0_ax1_fused_1_fused % T.int64(16)) * T.int64(2) + ax1_fused_2 < T.int64(8))
                                T.reads(C_pad_local[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_pad_local[v0, v1]
    # fmt: on

    mod = tvm.IRModule({"main": func})
    with Target("cuda"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4))(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_outer_reduction():
    # fmt: off
    @T.prim_func(private=True)
    def before(
        B0: T.Buffer((512, 6144), "uint32"),
        B1: T.Buffer((128, 6144), "float16"),
        var_A: T.handle,
        var_C: T.handle
    ):
        batch_size = T.int32()
        A = T.match_buffer(var_A, (batch_size, 1, 4096), "float16")
        C = T.match_buffer(var_C, (batch_size, 1, 6144), "float16")
        compute = T.alloc_buffer((4096, 6144), "float16")
        B = T.alloc_buffer((4096, 6144), "float16")
        for i0, i1 in T.grid(4096, 6144):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(B0[v_i0 // 8, v_i1], T.Cast("uint32", v_i0 % 8 * 4)), T.uint32(15)))
        for i0, i1 in T.grid(4096, 6144):
            with T.block("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                B[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * B1[v_i0 // 32, v_i1]
        for i0, i1, i2, k in T.grid(batch_size, 1, 6144, 4096):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

    @T.prim_func(private=True)
    def expected(B0: T.Buffer((512, 6144), "uint32"), B1: T.Buffer((128, 6144), "float16"), var_A: T.handle, var_C: T.handle):
        T.func_attr({"tir.is_scheduled": 1})
        batch_size = T.int32()
        A = T.match_buffer(var_A, (batch_size, 1, 4096), "float16")
        C = T.match_buffer(var_C, (batch_size, 1, 6144), "float16")
        # with T.block("root"):
        B_local = T.alloc_buffer((4096, 6144), "float16", scope="local")
        A_pad_shared = T.alloc_buffer(((batch_size + 3) // 4 * 4, 1, 4096), "float16", scope="shared")
        C_pad_local = T.alloc_buffer(((batch_size + 3) // 4 * 4, 1, 6144), "float16", scope="local")
        C_pad_rf_local = T.alloc_buffer((32, (batch_size + 3) // 4 * 4, 1, 6144), "float16", scope="local")
        C_pad_rf_local_1 = T.alloc_buffer((4, (batch_size + 3) // 4 * 4, 1, 6144), "float16", scope="local")
        B0_local = T.alloc_buffer((512, 6144), "uint32", scope="local")
        B1_local = T.alloc_buffer((128, 6144), "float16", scope="local")
        for ax0_0 in T.thread_binding((batch_size + 3) // 4, thread="blockIdx.y"):
            for ax1_fused_0 in T.thread_binding(96, thread="blockIdx.x"):
                for ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax2_fused_1_ax2_fused_3_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                        for ax0_1_init, ax2_fused_1_ax2_fused_3_fused_1_0_init in T.grid(4, 2):
                            for ax2_fused_1_ax2_fused_3_fused_1_1_init in T.vectorized(4):
                                with T.block("matmul_rf_init"):
                                    vax2_fused_1_ax2_fused_3_fused = T.axis.spatial(32, ax2_fused_1_ax2_fused_3_fused_0 * 8 + ax2_fused_1_ax2_fused_3_fused_1_0_init * 4 + ax2_fused_1_ax2_fused_3_fused_1_1_init)
                                    v0 = T.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax0_1_init)
                                    v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                    T.reads()
                                    T.writes(C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1])
                                    C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1] = T.float16(0)
                        for ax2_fused_0 in range(32):
                            for ax0_ax1_fused in T.vectorized(4):
                                with T.block("B0_local"):
                                    v0 = T.axis.spatial(512, ax2_fused_0 * 16 + ax2_fused_1_ax2_fused_3_fused_0 * 4 + ax0_ax1_fused)
                                    v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                    T.reads(B0[v0, v1])
                                    T.writes(B0_local[v0, v1])
                                    B0_local[v0, v1] = B0[v0, v1]
                            for ax0_ax1_fused in T.vectorized(1):
                                with T.block("B1_local"):
                                    v0 = T.axis.spatial(128, ax2_fused_0 * 4 + ax2_fused_1_ax2_fused_3_fused_0)
                                    v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                    T.reads(B1[v0, v1])
                                    T.writes(B1_local[v0, v1])
                                    B1_local[v0, v1] = B1[v0, v1]
                            for ax0_ax1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                    for ax0_ax1_fused_2 in T.vectorized(2):
                                        with T.block("A_pad"):
                                            v0 = T.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 128)
                                            v1 = T.axis.spatial(4096, ax2_fused_0 * 128 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 128)
                                            T.reads(A[v0, 0, v1])
                                            T.writes(A_pad_shared[v0, 0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 1, 8, 1]]})
                                            A_pad_shared[v0, 0, v1] = T.if_then_else(v0 < batch_size, A[v0, 0, v1], T.float16(0))
                            for ax2_fused_2 in range(4):
                                for ax0_ax1_fused_0 in range(2):
                                    for ax0_ax1_fused_1 in T.vectorized(4):
                                        with T.block("dequantize"):
                                            v0 = T.axis.spatial(4096, ax2_fused_0 * 128 + ax2_fused_1_ax2_fused_3_fused_0 * 32 + ax2_fused_2 * 8 + ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1)
                                            v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                            T.reads(B0_local[v0 // 8, v1], B1_local[v0 // 32, v1])
                                            T.writes(B_local[v0, v1])
                                            B_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(B0_local[v0 // 8, v1], T.Cast("uint32", v0 % 8 * 4)), T.uint32(15))) - T.float16(7)) * B1_local[v0 // 32, v1]
                                for ax0_1, ax2_fused_1_ax2_fused_3_fused_1_0 in T.grid(4, 2):
                                    for ax2_fused_1_ax2_fused_3_fused_1_1 in T.vectorized(4):
                                        with T.block("matmul_rf_update"):
                                            vax2_fused_1_ax2_fused_3_fused = T.axis.spatial(32, ax2_fused_1_ax2_fused_3_fused_0 * 8 + ax2_fused_1_ax2_fused_3_fused_1_0 * 4 + ax2_fused_1_ax2_fused_3_fused_1_1)
                                            v0 = T.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax0_1)
                                            v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                            vax2_fused_0, vax2_fused_2 = T.axis.remap("RR", [ax2_fused_0, ax2_fused_2])
                                            T.reads(C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1], A_pad_shared[v0, 0, vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8], B_local[vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8, v1])
                                            T.writes(C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1])
                                            C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1] = C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1] + A_pad_shared[v0, 0, vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8] * B_local[vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8, v1]
                for ax3 in T.thread_binding(64, thread="threadIdx.x"):
                    for ax0 in T.thread_binding(4, thread="threadIdx.y"):
                        for ax2_init in range(4):
                            with T.block("matmul_rf_init"):
                                vax2_fused_1_ax2_fused_3_fused_0 = T.axis.spatial(4, ax0)
                                v0 = T.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax2_init)
                                v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax3)
                                T.reads()
                                T.writes(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1])
                                C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1] = T.float16(0)
                        for ax2, ax1 in T.grid(4, 8):
                            with T.block("matmul_rf_update"):
                                vax2_fused_1_ax2_fused_3_fused_0, vax2_fused_1_ax2_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                v0 = T.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax2)
                                v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax3)
                                T.reads(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1], C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused_0 * 8 + vax2_fused_1_ax2_fused_3_fused_1, v0, 0, v1])
                                T.writes(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1])
                                C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1] = C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1] + C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused_0 * 8 + vax2_fused_1_ax2_fused_3_fused_1, v0, 0, v1]
                for ax1 in range(4):
                    for ax2 in T.thread_binding(64, thread="threadIdx.x"):
                        for ax0 in T.thread_binding(4, thread="threadIdx.y"):
                            with T.block("matmul"):
                                vax2_fused_1_ax2_fused_3_fused_0 = T.axis.reduce(4, ax0)
                                v0 = T.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax1)
                                v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax2)
                                T.reads(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1])
                                T.writes(C_pad_local[v0, 0, v1])
                                with T.init():
                                    C_pad_local[v0, 0, v1] = T.float16(0)
                                C_pad_local[v0, 0, v1] = C_pad_local[v0, 0, v1] + C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1]
                for ax0 in range(4):
                    for ax1 in T.thread_binding(64, thread="threadIdx.x"):
                        with T.block("C_pad"):
                            v0 = T.axis.spatial(batch_size, ax0_0 * 4 + ax0)
                            v1 = T.axis.spatial(6144, ax1_fused_0 * 64 + ax1)
                            T.where((ax0_0 - (batch_size + 3) // 4 < 0 or ax0_0 == 0) and ax0_0 * 4 + ax0 < batch_size)
                            T.reads(C_pad_local[v0, 0, v1])
                            T.writes(C[v0, 0, v1])
                            C[v0, 0, v1] = C_pad_local[v0, 0, v1]
    # fmt: on
    mod = tvm.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4))(mod)  # pylint: disable=not-callable
    tvm.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
