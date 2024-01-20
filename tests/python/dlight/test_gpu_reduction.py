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
# pylint: disable=missing-docstring,line-too-long,invalid-name,too-few-public-methods,too-many-locals

import tvm.testing
from tvm import dlight as dl
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


def test_decode_gemv_1():
    # NK layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            B = T.alloc_buffer((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    T.writes(C[v_i0, v_i1, v_i2])
                    with T.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_i2, v_k]


    @I.ir_module
    class After:
        @T.prim_func
        def func(W_handle: T.handle, S_handle: T.handle, V_handle: T.handle, C_handle: T.handle):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            W = T.match_buffer(W_handle, (4096, 512), "uint32")
            S = T.match_buffer(S_handle, (4096, 128), "float16")
            V = T.match_buffer(V_handle, (1, 1, 4096), "float16")
            C = T.match_buffer(C_handle, (1, 1, 4096), "float16")
            with T.block("root"):
                T.reads()
                T.writes()
                C_rf_local = T.alloc_buffer((512, 1, 1, 4096), "float16", scope="local")
                for ax0_fused in T.thread_binding(4096, thread="blockIdx.x"):
                    for ax1_0_fused_1 in T.thread_binding(512, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("matmul_rf_init"):
                            vax1_0_fused_1 = T.axis.spatial(512, ax1_0_fused_1)
                            v0 = T.axis.spatial(4096, ax0_fused)
                            T.reads()
                            T.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                            C_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_0_fused_0 in range(1):
                            for ax1_1 in range(8):
                                with T.block("matmul_rf_update"):
                                    vax1_0_fused_1 = T.axis.spatial(512, ax1_0_fused_1)
                                    v0 = T.axis.spatial(4096, ax0_fused)
                                    vax1_0_fused_0 = T.axis.reduce(1, ax1_0_fused_0)
                                    vax1_1 = T.axis.reduce(8, ax1_1)
                                    T.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0], V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1], W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                    T.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                    C_rf_local[vax1_0_fused_1, 0, 0, v0] = C_rf_local[vax1_0_fused_1, 0, 0, v0] + V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.thread_binding(512, thread="threadIdx.x"):
                            with T.block("matmul"):
                                vax1_0_fused_1 = T.axis.reduce(512, ax0)
                                v0 = T.axis.spatial(4096, ax0_fused)
                                T.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                T.writes(C[0, 0, v0])
                                with T.init():
                                    C[0, 0, v0] = T.float16(0)
                                C[0, 0, v0] = C[0, 0, v0] + C_rf_local[vax1_0_fused_1, 0, 0, v0]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_2():
    # KN layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def func(W: T.Buffer((512, 4096), "uint32"), S: T.Buffer((128, 4096), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            B = T.alloc_buffer((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(W[v_i // 8, v_j], S[v_i // 32, v_j])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i // 32, v_j]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(V[v_i0, v_i1, v_k], B[v_k, v_i2])
                    T.writes(C[v_i0, v_i1, v_i2])
                    with T.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_k, v_i2]


    @I.ir_module
    class After:
        @T.prim_func
        def func(W: T.Buffer((512, 4096), "uint32"), S: T.Buffer((128, 4096), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            C_rf_local = T.alloc_buffer((16, 1, 1, 4096), "float16", scope="local")
            for i2_i0_i1_fused_0 in T.thread_binding(256, thread="blockIdx.x"):
                for i2_i0_i1_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for k_0_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                        with T.block("matmul_rf_init"):
                            vk_0_fused_1 = T.axis.spatial(16, k_0_fused_1)
                            v_i2 = T.axis.spatial(4096, i2_i0_i1_fused_0 * 16 + i2_i0_i1_fused_1)
                            C_rf_local[vk_0_fused_1, 0, 0, v_i2] = T.float16(0)
                        for k_0_fused_0, k_1 in T.grid(32, 8):
                            with T.block("matmul_rf_update"):
                                vk_0_fused_1 = T.axis.spatial(16, k_0_fused_1)
                                v_i2 = T.axis.spatial(4096, i2_i0_i1_fused_0 * 16 + i2_i0_i1_fused_1)
                                vk_0_fused_0, vk_1 = T.axis.remap("RR", [k_0_fused_0, k_1])
                                C_rf_local[vk_0_fused_1, 0, 0, v_i2] = C_rf_local[vk_0_fused_1, 0, 0, v_i2] + V[0, 0, vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[(vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1) // 8, v_i2], T.Cast("uint32", (vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[(vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1) // 32, v_i2])
                for ax1_ax2_ax3_fused in T.thread_binding(16, thread="threadIdx.x"):
                    for ax0_fused in T.thread_binding(16, thread="threadIdx.y"):
                        with T.block("matmul"):
                            vk_0_fused_1 = T.axis.reduce(16, ax0_fused)
                            v_i2 = T.axis.spatial(4096, i2_i0_i1_fused_0 * 16 + ax1_ax2_ax3_fused)
                            with T.init():
                                C[0, 0, v_i2] = T.float16(0)
                            C[0, 0, v_i2] = C[0, 0, v_i2] + C_rf_local[vk_0_fused_1, 0, 0, v_i2]

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_3():
    # NK layout + N as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def func(W: T.Buffer((512, 4096), "uint32"), S: T.Buffer((128, 4096), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            B = T.alloc_buffer((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(W[v_i // 8, v_j], S[v_i // 32, v_j])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i // 32, v_j]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    T.writes(C[v_i0, v_i1, v_i2])
                    with T.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @I.ir_module
    class After:
        @T.prim_func
        def func(W_handle: T.handle, S_handle: T.handle, V_handle: T.handle, C_handle: T.handle):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            W = T.match_buffer(W_handle, (512, 4096), "uint32")
            S = T.match_buffer(S_handle, (128, 4096), "float16")
            V = T.match_buffer(V_handle, (1, 1, 4096), "float16")
            C = T.match_buffer(C_handle, (1, 1, 4096), "float16")
            with T.block("root"):
                T.reads()
                T.writes()
                C_rf_local = T.alloc_buffer((1024, 1, 1, 4096), "float16", scope="local")
                for ax0_0_fused in T.thread_binding(512, thread="blockIdx.x"):
                    for ax1_fused_1 in T.thread_binding(1024, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        for ax0_1_init in range(8):
                            with T.block("matmul_rf_init"):
                                vax1_fused_1 = T.axis.spatial(1024, ax1_fused_1)
                                v0 = T.axis.spatial(4096, ax0_0_fused * 8 + ax0_1_init)
                                T.reads()
                                T.writes(C_rf_local[vax1_fused_1, 0, 0, v0])
                                C_rf_local[vax1_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_fused_0 in range(4):
                            for ax0_1 in range(8):
                                with T.block("matmul_rf_update"):
                                    vax1_fused_1 = T.axis.spatial(1024, ax1_fused_1)
                                    v0 = T.axis.spatial(4096, ax0_0_fused * 8 + ax0_1)
                                    vax1_fused_0 = T.axis.reduce(4, ax1_fused_0)
                                    T.reads(C_rf_local[vax1_fused_1, 0, 0, v0], V[0, 0, vax1_fused_0 * 1024 + vax1_fused_1], W[v0 // 8, vax1_fused_0 * 1024 + vax1_fused_1], S[v0 // 32, vax1_fused_0 * 1024 + vax1_fused_1])
                                    T.writes(C_rf_local[vax1_fused_1, 0, 0, v0])
                                    C_rf_local[vax1_fused_1, 0, 0, v0] = C_rf_local[vax1_fused_1, 0, 0, v0] + V[0, 0, vax1_fused_0 * 1024 + vax1_fused_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v0 // 8, vax1_fused_0 * 1024 + vax1_fused_1], T.Cast("uint32", v0 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0 // 32, vax1_fused_0 * 1024 + vax1_fused_1])
                    for ax1_fused_0 in range(1):
                        for ax0 in T.thread_binding(1024, thread="threadIdx.x"):
                            for ax1_fused_1 in range(8):
                                with T.block("matmul"):
                                    vax1_fused_1 = T.axis.reduce(1024, ax0)
                                    v0 = T.axis.spatial(4096, ax0_0_fused * 8 + ax1_fused_1)
                                    T.reads(C_rf_local[vax1_fused_1, 0, 0, v0])
                                    T.writes(C[0, 0, v0])
                                    with T.init():
                                        C[0, 0, v0] = T.float16(0)
                                    C[0, 0, v0] = C[0, 0, v0] + C_rf_local[vax1_fused_1, 0, 0, v0]

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_4():
    # KN layout + N as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            B = T.alloc_buffer((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(V[v_i0, v_i1, v_k], B[v_k, v_i2])
                    T.writes(C[v_i0, v_i1, v_i2])
                    with T.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_k, v_i2]


    @I.ir_module
    class After:
        @T.prim_func
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            C_rf_local = T.alloc_buffer((16, 1, 1, 4096), "float16", scope="local")
            for i2_0_i0_i1_fused_0 in T.thread_binding(32, thread="blockIdx.x"):
                for i2_0_i0_i1_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for k_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                        for i2_1_init in range(8):
                            with T.block("matmul_rf_init"):
                                vk_fused_1 = T.axis.spatial(16, k_fused_1)
                                v_i2 = T.axis.spatial(4096, i2_0_i0_i1_fused_0 * 128 + i2_0_i0_i1_fused_1 * 8 + i2_1_init)
                                C_rf_local[vk_fused_1, 0, 0, v_i2] = T.float16(0)
                        for k_fused_0, i2_1 in T.grid(256, 8):
                            with T.block("matmul_rf_update"):
                                vk_fused_1 = T.axis.spatial(16, k_fused_1)
                                v_i2 = T.axis.spatial(4096, i2_0_i0_i1_fused_0 * 128 + i2_0_i0_i1_fused_1 * 8 + i2_1)
                                vk_fused_0 = T.axis.reduce(256, k_fused_0)
                                C_rf_local[vk_fused_1, 0, 0, v_i2] = C_rf_local[vk_fused_1, 0, 0, v_i2] + V[0, 0, vk_fused_0 * 16 + vk_fused_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[vk_fused_0 * 16 + vk_fused_1, v_i2 // 8], T.Cast("uint32", v_i2 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[vk_fused_0 * 16 + vk_fused_1, v_i2 // 32])
                for ax1_ax2_ax3_fused_0 in T.thread_binding(16, thread="threadIdx.x"):
                    for ax1_ax2_ax3_fused_1 in range(8):
                        for ax0_fused in T.thread_binding(16, thread="threadIdx.y"):
                            with T.block("matmul"):
                                vk_fused_1 = T.axis.reduce(16, ax0_fused)
                                v_i2 = T.axis.spatial(4096, i2_0_i0_i1_fused_0 * 128 + ax1_ax2_ax3_fused_0 * 8 + ax1_ax2_ax3_fused_1)
                                with T.init():
                                    C[0, 0, v_i2] = T.float16(0)
                                C[0, 0, v_i2] = C[0, 0, v_i2] + C_rf_local[vk_fused_1, 0, 0, v_i2]

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_sigmoid():
    # NK layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), D: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            B = T.alloc_buffer((4096, 4096), "float16")
            C = T.alloc_buffer((1, 1, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    T.writes(C[v_i0, v_i1, v_i2])
                    with T.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_i2, v_k]
            for i0, i1, i2 in T.grid(1, 1, 4096):
                with T.block("sigmoid"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(C[v_i0, v_i1, v_i2])
                    T.writes(D[v_i0, v_i1, v_i2])
                    D[v_i0, v_i1, v_i2] = T.sigmoid(C[v_i0, v_i1, v_i2])

    @I.ir_module
    class After:
        @T.prim_func
        def func(W_handle: T.handle, S_handle: T.handle, V_handle: T.handle, D_handle: T.handle):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            W = T.match_buffer(W_handle, (4096, 512), "uint32")
            S = T.match_buffer(S_handle, (4096, 128), "float16")
            V = T.match_buffer(V_handle, (1, 1, 4096), "float16")
            D = T.match_buffer(D_handle, (1, 1, 4096), "float16")
            with T.block("root"):
                T.reads()
                T.writes()
                C_local = T.alloc_buffer((1, 1, 4096), "float16", scope="local")
                C_rf_local = T.alloc_buffer((512, 1, 1, 4096), "float16", scope="local")
                for ax0_fused in T.thread_binding(4096, thread="blockIdx.x"):
                    for ax1_0_fused_1 in T.thread_binding(512, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("matmul_rf_init"):
                            vax1_0_fused_1 = T.axis.spatial(512, ax1_0_fused_1)
                            v0 = T.axis.spatial(4096, ax0_fused)
                            T.reads()
                            T.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                            C_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_0_fused_0 in range(1):
                            for ax1_1 in range(8):
                                with T.block("matmul_rf_update"):
                                    vax1_0_fused_1 = T.axis.spatial(512, ax1_0_fused_1)
                                    v0 = T.axis.spatial(4096, ax0_fused)
                                    vax1_0_fused_0 = T.axis.reduce(1, ax1_0_fused_0)
                                    vax1_1 = T.axis.reduce(8, ax1_1)
                                    T.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0], V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1], W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                    T.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                    C_rf_local[vax1_0_fused_1, 0, 0, v0] = C_rf_local[vax1_0_fused_1, 0, 0, v0] + V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.thread_binding(512, thread="threadIdx.x"):
                            with T.block("matmul"):
                                vax1_0_fused_1 = T.axis.reduce(512, ax0)
                                v0 = T.axis.spatial(4096, ax0_fused)
                                T.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                T.writes(C_local[0, 0, v0])
                                with T.init():
                                    C_local[0, 0, v0] = T.float16(0)
                                C_local[0, 0, v0] = C_local[0, 0, v0] + C_rf_local[vax1_0_fused_1, 0, 0, v0]
                    with T.block("sigmoid"):
                        v0 = T.axis.spatial(4096, ax0_fused)
                        T.reads(C_local[0, 0, v0])
                        T.writes(D[0, 0, v0])
                        D[0, 0, v0] = T.sigmoid(C_local[0, 0, v0])

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_1_fp32():
    # NK layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            B = T.alloc_buffer((4096, 4096), "float16")
            C_fp32 = T.alloc_buffer((1, 1, 4096), "float32")
            for i, j in T.grid(4096, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    T.writes(C_fp32[v_i0, v_i1, v_i2])
                    with T.init():
                        C_fp32[v_i0, v_i1, v_i2] = T.float16(0)
                    C_fp32[v_i0, v_i1, v_i2] = C_fp32[v_i0, v_i1, v_i2] + T.Cast("float32", V[v_i0, v_i1, v_k]) * T.Cast("float32", B[v_i2, v_k])
            for i0, i1, i2 in T.grid(1, 1, 4096):
                with T.block("cast"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(C_fp32[v_i0, v_i1, v_i2])
                    T.writes(C[v_i0, v_i1, v_i2])
                    C[v_i0, v_i1, v_i2] = T.Cast("float16", C_fp32[v_i0, v_i1, v_i2])

    @I.ir_module
    class After:
        @T.prim_func
        def func(W_handle: T.handle, S_handle: T.handle, V_handle: T.handle, C_handle: T.handle):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            W = T.match_buffer(W_handle, (4096, 512), "uint32")
            S = T.match_buffer(S_handle, (4096, 128), "float16")
            V = T.match_buffer(V_handle, (1, 1, 4096), "float16")
            C = T.match_buffer(C_handle, (1, 1, 4096), "float16")
            with T.block("root"):
                T.reads()
                T.writes()
                C_fp32_local = T.alloc_buffer((1, 1, 4096), scope="local")
                C_fp32_rf_local = T.alloc_buffer((512, 1, 1, 4096), scope="local")
                for ax0_fused in T.thread_binding(4096, thread="blockIdx.x"):
                    for ax1_0_fused_1 in T.thread_binding(512, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("matmul_rf_init"):
                            vax1_0_fused_1 = T.axis.spatial(512, ax1_0_fused_1)
                            v0 = T.axis.spatial(4096, ax0_fused)
                            T.reads()
                            T.writes(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                            C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float32(0)
                        for ax1_0_fused_0 in range(1):
                            for ax1_1 in range(8):
                                with T.block("matmul_rf_update"):
                                    vax1_0_fused_1 = T.axis.spatial(512, ax1_0_fused_1)
                                    v0 = T.axis.spatial(4096, ax0_fused)
                                    vax1_0_fused_0 = T.axis.reduce(1, ax1_0_fused_0)
                                    vax1_1 = T.axis.reduce(8, ax1_1)
                                    T.reads(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0], V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1], W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                    T.writes(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                                    C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] = C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] + T.Cast("float32", V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1]) * T.Cast("float32", (T.Cast("float16", T.bitwise_and(T.shift_right(W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.thread_binding(512, thread="threadIdx.x"):
                            with T.block("matmul"):
                                vax1_0_fused_1 = T.axis.reduce(512, ax0)
                                v0 = T.axis.spatial(4096, ax0_fused)
                                T.reads(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                                T.writes(C_fp32_local[0, 0, v0])
                                with T.init():
                                    C_fp32_local[0, 0, v0] = T.float32(0)
                                C_fp32_local[0, 0, v0] = C_fp32_local[0, 0, v0] + C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0]
                    with T.block("cast"):
                        v0 = T.axis.spatial(4096, ax0_fused)
                        T.reads(C_fp32_local[0, 0, v0])
                        T.writes(C[0, 0, v0])
                        C[0, 0, v0] = T.Cast("float16", C_fp32_local[0, 0, v0])

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_reduction_no_spatial():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((1, 1, 4096), "float16"), B: T.Buffer((4096,), "float16"), rms_norm: T.Buffer((1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            Ared_temp = T.alloc_buffer((1, 1))
            for ax0 in range(4096):
                with T.block("Ared_temp"):
                    v0 = T.axis.reduce(4096, ax0)
                    with T.init():
                        Ared_temp[0, 0] = T.float32(0)
                    Ared_temp[0, 0] = Ared_temp[0, 0] + T.Cast("float32", A[0, 0, v0]) * T.Cast("float32", A[0, 0, v0])
            for ax0 in range(4096):
                with T.block("rms_norm"):
                    v0 = T.axis.spatial(4096, ax0)
                    rms_norm[0, v0] = T.Cast("float16", T.Cast("float32", B[v0]) * (T.Cast("float32", A[0, 0, v0]) / T.sqrt(Ared_temp[0, 0] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

    @I.ir_module
    class After:
        @T.prim_func
        def main(A_handle: T.handle, B_handle: T.handle, rms_norm_handle: T.handle):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            A = T.match_buffer(A_handle, (1, 1, 4096), "float16")
            B = T.match_buffer(B_handle, (4096,), "float16")
            rms_norm = T.match_buffer(rms_norm_handle, (1, 4096), "float16")
            with T.block("root"):
                T.reads()
                T.writes()
                Ared_temp_shared = T.alloc_buffer((1, 1), scope="shared")
                Ared_temp_rf_local = T.alloc_buffer((1024, 1, 1), scope="local")
                for ax0_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                    for ax1_fused_1 in T.thread_binding(1024, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("Ared_temp_rf_init"):
                            vax1_fused_1 = T.axis.spatial(1024, ax1_fused_1)
                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                            T.reads()
                            T.writes(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                            Ared_temp_rf_local[vax1_fused_1, 0, 0] = T.float32(0)
                        for ax1_fused_0 in range(4):
                            for u in range(1):
                                with T.block("Ared_temp_rf_update"):
                                    vax1_fused_1 = T.axis.spatial(1024, ax1_fused_1)
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    vax1_fused_0 = T.axis.reduce(4, ax1_fused_0)
                                    T.reads(Ared_temp_rf_local[vax1_fused_1, 0, 0], A[0, 0, vax1_fused_0 * 1024 + vax1_fused_1])
                                    T.writes(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                                    Ared_temp_rf_local[vax1_fused_1, 0, 0] = Ared_temp_rf_local[vax1_fused_1, 0, 0] + T.Cast("float32", A[0, 0, vax1_fused_0 * 1024 + vax1_fused_1]) * T.Cast("float32", A[0, 0, vax1_fused_0 * 1024 + vax1_fused_1])
                    for ax1_fused in range(T.int64(1)):
                        for ax0 in T.thread_binding(1024, thread="threadIdx.x"):
                            with T.block("Ared_temp"):
                                vax1_fused_1 = T.axis.reduce(1024, ax0)
                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                                T.writes(Ared_temp_shared[0, 0])
                                with T.init():
                                    Ared_temp_shared[0, 0] = T.float32(0)
                                Ared_temp_shared[0, 0] = Ared_temp_shared[0, 0] + Ared_temp_rf_local[vax1_fused_1, 0, 0]
                    for ax0_fused_0 in range(4):
                        for ax0_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                            with T.block("rms_norm"):
                                v0 = T.axis.spatial(4096, ax0_fused_0 * 1024 + ax0_fused_1)
                                T.reads(B[v0], A[0, 0, v0], Ared_temp_shared[0, 0])
                                T.writes(rms_norm[0, v0])
                                rms_norm[0, v0] = T.Cast("float16", T.Cast("float32", B[v0]) * (T.Cast("float32", A[0, 0, v0]) / T.sqrt(Ared_temp_shared[0, 0] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
    # fmt: on
    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_spatial_inner_no_broadcasting():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def main(lv575: T.Buffer((1376, 4096), "uint32"), lv576: T.Buffer((344, 4096), "float16"), lv574: T.Buffer((1, 1, 11008), "float16"), lv570: T.Buffer((1, 1, 4096), "float16"), p_output0_intermediate: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"tir.noalias": T.bool(True)})
            p_output0_intermediate_1 = T.alloc_buffer((11008, 4096), "float16")
            var_matmul_intermediate = T.alloc_buffer((1, 1, 4096), "float16")
            for i, j in T.grid(11008, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(lv575[v_i // 8, v_j], lv576[v_i // 32, v_j])
                    T.writes(p_output0_intermediate_1[v_i, v_j])
                    p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[v_i // 32, v_j]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 11008):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(lv574[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                    T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                    with T.init():
                        var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv574[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
            for ax0, ax1, ax2 in T.grid(1, 1, 4096):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(lv570[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                    T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                    p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv570[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(lv575: T.Buffer((1376, 4096), "uint32"), lv576: T.Buffer((344, 4096), "float16"), lv574: T.Buffer((1, 1, 11008), "float16"), lv570: T.Buffer((1, 1, 4096), "float16"), p_output0_intermediate: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            var_matmul_intermediate_local = T.alloc_buffer((1, 1, 4096), "float16", scope="local")
            var_matmul_intermediate_rf_local = T.alloc_buffer((16, 1, 1, 4096), "float16", scope="local")
            for ax0_fused_0 in T.thread_binding(256, thread="blockIdx.x"):
                for ax0_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for ax1_0_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                        with T.block("matmul_rf_init"):
                            vax1_0_fused_1 = T.axis.spatial(16, ax1_0_fused_1)
                            v0 = T.axis.spatial(4096, ax0_fused_0 * 16 + ax0_fused_1)
                            T.reads()
                            T.writes(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                            var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_0_fused_0, ax1_1 in T.grid(86, 8):
                            with T.block("matmul_rf_update"):
                                vax1_0_fused_1 = T.axis.spatial(16, ax1_0_fused_1)
                                v0 = T.axis.spatial(4096, ax0_fused_0 * 16 + ax0_fused_1)
                                vax1_0_fused_0, vax1_1 = T.axis.remap("RR", [ax1_0_fused_0, ax1_1])
                                T.reads(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0], lv574[0, 0, vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1], lv575[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 8, v0], lv576[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 32, v0])
                                T.writes(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] + lv574[0, 0, vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv575[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 8, v0], T.Cast("uint32", (vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 32, v0])
                for ax1_fused in T.thread_binding(16, thread="threadIdx.x"):
                    for ax0 in T.thread_binding(16, thread="threadIdx.y"):
                        with T.block("matmul"):
                            vax1_0_fused_1 = T.axis.reduce(16, ax0)
                            v0 = T.axis.spatial(4096, ax0_fused_0 * 16 + ax1_fused)
                            T.reads(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                            T.writes(var_matmul_intermediate_local[0, 0, v0])
                            with T.init():
                                var_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                            var_matmul_intermediate_local[0, 0, v0] = var_matmul_intermediate_local[0, 0, v0] + var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0]
                for ax0_fused_0_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for ax0_fused_1 in range(1):
                        with T.block("T_add"):
                            v0 = T.axis.spatial(4096, ax0_fused_0 * 16 + ax0_fused_0_1 + ax0_fused_1)
                            T.reads(lv570[0, 0, v0], var_matmul_intermediate_local[0, 0, v0])
                            T.writes(p_output0_intermediate[0, 0, v0])
                            p_output0_intermediate[0, 0, v0] = lv570[0, 0, v0] + var_matmul_intermediate_local[0, 0, v0]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_spatial_inner_broadcasting():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256, 256), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            temp_local = T.alloc_buffer((256,))
            for j in T.serial(256):
                for k in T.serial(256):
                    with T.block("sum"):
                        vj, vk = T.axis.remap("SR", [j, k])
                        T.reads(A[vk, vj])
                        T.writes(temp_local[vj])
                        with T.init():
                            temp_local[vj] = T.float32(0)
                        temp_local[vj] = temp_local[vj] + A[vk, vj]
            for i, j in T.grid(256, 256):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.reads(temp_local[vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + temp_local[vj]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256, 256), "float32")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            temp_local_shared = T.alloc_buffer((256,), scope="shared")
            temp_local_rf_local = T.alloc_buffer((16, 256), scope="local")
            for ax0_fused_0 in T.thread_binding(16, thread="blockIdx.x"):
                for ax0_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                        with T.block("sum_rf_init"):
                            vax1_fused_1 = T.axis.spatial(16, ax1_fused_1)
                            v0 = T.axis.spatial(256, ax0_fused_0 * 16 + ax0_fused_1)
                            T.reads()
                            T.writes(temp_local_rf_local[vax1_fused_1, v0])
                            temp_local_rf_local[vax1_fused_1, v0] = T.float32(0)
                        for ax1_fused_0, u in T.grid(16, 1):
                            with T.block("sum_rf_update"):
                                vax1_fused_1 = T.axis.spatial(16, ax1_fused_1)
                                v0 = T.axis.spatial(256, ax0_fused_0 * 16 + ax0_fused_1)
                                vax1_fused_0 = T.axis.reduce(16, ax1_fused_0)
                                T.reads(temp_local_rf_local[vax1_fused_1, v0], A[vax1_fused_0 * 16 + vax1_fused_1, v0])
                                T.writes(temp_local_rf_local[vax1_fused_1, v0])
                                temp_local_rf_local[vax1_fused_1, v0] = temp_local_rf_local[vax1_fused_1, v0] + A[vax1_fused_0 * 16 + vax1_fused_1, v0]
                for ax1_fused in T.thread_binding(16, thread="threadIdx.x"):
                    for ax0 in T.thread_binding(16, thread="threadIdx.y"):
                        with T.block("sum"):
                            vax1_fused_1 = T.axis.reduce(16, ax0)
                            v0 = T.axis.spatial(256, ax0_fused_0 * 16 + ax1_fused)
                            T.reads(temp_local_rf_local[vax1_fused_1, v0])
                            T.writes(temp_local_shared[v0])
                            with T.init():
                                temp_local_shared[v0] = T.float32(0)
                            temp_local_shared[v0] = temp_local_shared[v0] + temp_local_rf_local[vax1_fused_1, v0]
                for ax0_ax1_fused_0 in range(16):
                    for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                        for ax0_ax1_fused_2 in T.thread_binding(16, thread="threadIdx.y"):
                            with T.block("add"):
                                v0 = T.axis.spatial(256, (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) // 16)
                                v1 = T.axis.spatial(256, ax0_fused_0 * 16 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) % 16)
                                T.reads(temp_local_shared[v1])
                                T.writes(B[v0, v1])
                                B[v0, v1] = A[v0, v1] + temp_local_shared[v1]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_reduction_inner_no_broadcasting():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256,), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            temp_local = T.alloc_buffer((256,))
            for i in T.serial(256):
                for k in T.serial(256):
                    with T.block("sum"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        T.reads(A[vi, vk])
                        T.writes(temp_local[vi])
                        with T.init():
                            temp_local[vi] = T.float32(0)
                        temp_local[vi] = temp_local[vi] + A[vi, vk]
            for i in T.grid(256):
                with T.block("add"):
                    vi = T.axis.remap("S", [i])
                    T.reads(temp_local[vi])
                    T.writes(B[vi,])
                    B[vi] = temp_local[vi] + T.float32(1)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256,), "float32")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            temp_local_local = T.alloc_buffer((256,), scope="local")
            temp_local_rf_local = T.alloc_buffer((256, 256), scope="local")
            for ax0_fused in T.thread_binding(256, thread="blockIdx.x"):
                for ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("sum_rf_init"):
                        vax1_fused_1, v0 = T.axis.remap("SS", [ax1_fused_1, ax0_fused])
                        T.reads()
                        T.writes(temp_local_rf_local[vax1_fused_1, v0])
                        temp_local_rf_local[vax1_fused_1, v0] = T.float32(0)
                    for ax1_fused_0, u in T.grid(1, 1):
                        with T.block("sum_rf_update"):
                            vax1_fused_1, v0, vax1_fused_0 = T.axis.remap("SSR", [ax1_fused_1, ax0_fused, ax1_fused_0])
                            T.reads(temp_local_rf_local[vax1_fused_1, v0], A[v0, vax1_fused_0 * 256 + vax1_fused_1])
                            T.writes(temp_local_rf_local[vax1_fused_1, v0])
                            temp_local_rf_local[vax1_fused_1, v0] = temp_local_rf_local[vax1_fused_1, v0] + A[v0, vax1_fused_0 * 256 + vax1_fused_1]
                for ax1_fused in range(1):
                    for ax0 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("sum"):
                            vax1_fused_1, v0 = T.axis.remap("RS", [ax0, ax0_fused])
                            T.reads(temp_local_rf_local[vax1_fused_1, v0])
                            T.writes(temp_local_local[v0])
                            with T.init():
                                temp_local_local[v0] = T.float32(0)
                            temp_local_local[v0] = temp_local_local[v0] + temp_local_rf_local[vax1_fused_1, v0]
                with T.block("add"):
                    v0 = T.axis.spatial(256, ax0_fused)
                    T.reads(temp_local_local[v0])
                    T.writes(B[v0])
                    B[v0] = temp_local_local[v0] + T.float32(1)
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_reduction_inner_no_broadcasting2():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def main(lv9: T.Buffer((2560, 320), "uint32"), lv10: T.Buffer((2560, 80), "float16"), lv1: T.Buffer((1, 2560), "float16"), p_output0_intermediate: T.Buffer((1, 2560), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            p_output0_intermediate_1 = T.alloc_buffer((2560, 2560), "float16")
            var_matmul_intermediate = T.alloc_buffer((1, 2560), "float16")
            for i, j in T.grid(2560, 2560):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(lv9[v_i, v_j // 8], lv10[v_i, v_j // 32])
                    T.writes(p_output0_intermediate_1[v_i, v_j])
                    p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv9[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv10[v_i, v_j // 32]
            for i0, i1, k in T.grid(1, 2560, 2560):
                with T.block("matmul"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(lv1[v_i0, v_k], p_output0_intermediate_1[v_k, v_i1])
                    T.writes(var_matmul_intermediate[v_i0, v_i1])
                    with T.init():
                        var_matmul_intermediate[v_i0, v_i1] = T.float16(0)
                    var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv1[v_i0, v_k] * p_output0_intermediate_1[v_k, v_i1]
            for i0, i1 in T.grid(1, 2560):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(var_matmul_intermediate[v_i0, v_i1])
                    T.writes(p_output0_intermediate[v_i0, v_i1])
                    p_output0_intermediate[v_i0, v_i1] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1])

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(lv9: T.Buffer((2560, 320), "uint32"), lv10: T.Buffer((2560, 80), "float16"), lv1: T.Buffer((1, 2560), "float16"), p_output0_intermediate: T.Buffer((1, 2560), "float32")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            var_matmul_intermediate_local = T.alloc_buffer((1, 2560), "float16", scope="local")
            var_matmul_intermediate_rf_local = T.alloc_buffer((16, 1, 2560), "float16", scope="local")
            for ax0_0_fused_0 in T.thread_binding(20, thread="blockIdx.x"):
                for ax0_0_fused_1 in T.thread_binding(16, thread="threadIdx.x"):
                    for ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                        for ax0_1_init in range(8):
                            with T.block("matmul_rf_init"):
                                vax1_fused_1 = T.axis.spatial(16, ax1_fused_1)
                                v0 = T.axis.spatial(2560, ax0_0_fused_0 * 128 + ax0_0_fused_1 * 8 + ax0_1_init)
                                T.reads()
                                T.writes(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0])
                                var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0] = T.float16(0)
                        for ax1_fused_0, ax0_1 in T.grid(160, 8):
                            with T.block("matmul_rf_update"):
                                vax1_fused_1 = T.axis.spatial(16, ax1_fused_1)
                                v0 = T.axis.spatial(2560, ax0_0_fused_0 * 128 + ax0_0_fused_1 * 8 + ax0_1)
                                vax1_fused_0 = T.axis.reduce(160, ax1_fused_0)
                                T.reads(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0], lv1[0, vax1_fused_0 * 16 + vax1_fused_1], lv9[vax1_fused_0 * 16 + vax1_fused_1, v0 // 8], lv10[vax1_fused_0 * 16 + vax1_fused_1, v0 // 32])
                                T.writes(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0])
                                var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0] = var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0] + lv1[0, vax1_fused_0 * 16 + vax1_fused_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv9[vax1_fused_0 * 16 + vax1_fused_1, v0 // 8], T.Cast("uint32", v0 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv10[vax1_fused_0 * 16 + vax1_fused_1, v0 // 32])
                for ax1_fused_0 in T.thread_binding(16, thread="threadIdx.x"):
                    for ax1_fused_1 in range(8):
                        for ax0 in T.thread_binding(16, thread="threadIdx.y"):
                            with T.block("matmul"):
                                vax1_fused_1 = T.axis.reduce(16, ax0)
                                v0 = T.axis.spatial(2560, ax0_0_fused_0 * 128 + ax1_fused_0 * 8 + ax1_fused_1)
                                T.reads(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0])
                                T.writes(var_matmul_intermediate_local[0, v0])
                                with T.init():
                                    var_matmul_intermediate_local[0, v0] = T.float16(0)
                                var_matmul_intermediate_local[0, v0] = var_matmul_intermediate_local[0, v0] + var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0]
                for ax0_fused_0 in T.thread_binding(16, thread="threadIdx.x"):
                    for ax0_fused_1 in range(8):
                        with T.block("compute"):
                            v0 = T.axis.spatial(2560, ax0_0_fused_0 * 128 + ax0_fused_0 * 8 + ax0_fused_1)
                            T.reads(var_matmul_intermediate_local[0, v0])
                            T.writes(p_output0_intermediate[0, v0])
                            p_output0_intermediate[0, v0] = T.Cast("float32", var_matmul_intermediate_local[0, v0])
    # fmt: on

    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_reduction_inner_spatial_choose_perfect_factor():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(100)), "float16")):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
            B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, T.int64(100)), "float16")
            # with T.block("root"):
            for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(100), n):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                    with T.init():
                        matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
    @I.ir_module
    class Expected:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(100)), "float16")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
            B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, T.int64(100)), "float16")
            # with T.block("root"):
            matmul_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(32), T.int64(1), T.int64(100)), "float16", scope="local")
            for ax0_ax1_fused_0 in T.thread_binding(T.int64(320), thread="blockIdx.x"):
                for ax0_ax1_fused_1 in T.thread_binding(T.int64(10), thread="threadIdx.x"):
                    for ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        with T.block("matmul_rf_init"):
                            vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
                            v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) // T.int64(100))
                            v1 = T.axis.spatial(T.int64(100), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) % T.int64(100))
                            T.reads()
                            T.writes(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                            matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_0, u in T.grid((n + T.int64(15)) // T.int64(16), 1):
                            with T.block("matmul_rf_update"):
                                vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
                                v0 = T.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) // T.int64(100))
                                v1 = T.axis.spatial(T.int64(100), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) % T.int64(100))
                                vax2_fused_0 = T.axis.reduce((n + T.int64(15)) // T.int64(16), ax2_fused_0)
                                T.where(ax2_fused_0 * T.int64(16) + ax2_fused_1 < n)
                                T.reads(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1], A[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1], B[T.int64(0), v0, vax2_fused_0 * T.int64(16) + vax2_fused_1, v1])
                                T.writes(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] + A[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1] * B[T.int64(0), v0, vax2_fused_0 * T.int64(16) + vax2_fused_1, v1]
                for ax1_ax2_fused in T.thread_binding(T.int64(10), thread="threadIdx.x"):
                    for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        with T.block("matmul"):
                            vax2_fused_1 = T.axis.reduce(T.int64(16), ax0)
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused_0 // T.int64(10))
                            v1 = T.axis.spatial(T.int64(100), ax0_ax1_fused_0 % T.int64(10) * T.int64(10) + ax1_ax2_fused)
                            T.reads(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                            T.writes(matmul[T.int64(0), v0, T.int64(0), v1])
                            with T.init():
                                matmul[T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                            matmul[T.int64(0), v0, T.int64(0), v1] = matmul[T.int64(0), v0, T.int64(0), v1] + matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1]
    # fmt: on

    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_repeat_transpose_gemv():
    # fmt: off

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def fused_relax_repeat_relax_permute_dims_relax_matmul1(p_lv716: T.handle, p_astype66: T.handle, var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
            T.func_attr({"tir.noalias": T.bool(True)})
            kv_seq_len = T.int64()
            lv716 = T.match_buffer(p_lv716, (T.int64(1), kv_seq_len, T.int64(8), T.int64(128)), "float16")
            astype66 = T.match_buffer(p_astype66, (T.int64(1), T.int64(32), T.int64(1), kv_seq_len), "float16")
            # with T.block("root"):
            var_T_repeat_intermediate = T.alloc_buffer((T.int64(1), kv_seq_len, T.int64(32), T.int64(128)), "float16")
            var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), kv_seq_len, T.int64(128)), "float16")
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), kv_seq_len, T.int64(32), T.int64(128)):
                with T.block("T_repeat"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(lv716[v_ax0, v_ax1, v_ax2 // T.int64(4), v_ax3])
                    T.writes(var_T_repeat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    var_T_repeat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv716[v_ax0, v_ax1, v_ax2 // T.int64(4), v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), kv_seq_len, T.int64(128)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(var_T_repeat_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
                    T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_repeat_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
            for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), kv_seq_len):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    T.reads(astype66[v_i0, v_i1, v_i2, v_k], var_T_transpose_intermediate[v_i0, v_i1, v_k, v_i3])
                    T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                    with T.init():
                        var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + astype66[v_i0, v_i1, v_i2, v_k] * var_T_transpose_intermediate[v_i0, v_i1, v_k, v_i3]
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused_relax_repeat_relax_permute_dims_relax_matmul1(p_lv716: T.handle, p_astype66: T.handle, var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            kv_seq_len = T.int64()
            lv716 = T.match_buffer(p_lv716, (T.int64(1), kv_seq_len, T.int64(8), T.int64(128)), "float16")
            astype66 = T.match_buffer(p_astype66, (T.int64(1), T.int64(32), T.int64(1), kv_seq_len), "float16")
            # with T.block("root"):
            var_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16", scope="local")
            for ax0_0_ax1_fused_0 in T.thread_binding(T.int64(64), thread="blockIdx.x"):
                for ax0_0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                    for ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        for ax0_1_init in range(T.int64(4)):
                            with T.block("matmul_rf_init"):
                                vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
                                v0 = T.axis.spatial(T.int64(32), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) // T.int64(128) * T.int64(4) + ax0_1_init)
                                v1 = T.axis.spatial(T.int64(128), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) % T.int64(128))
                                T.reads()
                                T.writes(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_0, ax0_1 in T.grid((kv_seq_len + T.int64(15)) // T.int64(16), T.int64(4)):
                            with T.block("matmul_rf_update"):
                                vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
                                v0 = T.axis.spatial(T.int64(32), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) // T.int64(128) * T.int64(4) + ax0_1)
                                v1 = T.axis.spatial(T.int64(128), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) % T.int64(128))
                                vax2_fused_0 = T.axis.reduce((kv_seq_len + T.int64(15)) // T.int64(16), ax2_fused_0)
                                T.where(ax2_fused_0 * T.int64(16) + ax2_fused_1 < kv_seq_len)
                                T.reads(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1], astype66[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1], lv716[T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1, v0 // T.int64(4), v1])
                                T.writes(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] + astype66[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1] * lv716[T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1, v0 // T.int64(4), v1]
                for ax1_0_ax2_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                    for ax1_1 in range(T.int64(4)):
                        for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            with T.block("matmul"):
                                vax2_fused_1 = T.axis.reduce(T.int64(16), ax0)
                                v0 = T.axis.spatial(T.int64(32), ax0_0_ax1_fused_0 // T.int64(8) * T.int64(4) + ax1_1)
                                v1 = T.axis.spatial(T.int64(128), ax0_0_ax1_fused_0 % T.int64(8) * T.int64(16) + ax1_0_ax2_fused)
                                T.reads(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                T.writes(var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1])
                                with T.init():
                                    var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                                var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1] = var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1] + var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1]
    # fmt: on

    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_gemv_dyn_shape_epilogue():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(
            var_A: T.handle,
            B: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
            var_C: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            vocab_size = T.int64()
            A = T.match_buffer(var_A, (T.int64(4096), vocab_size), "float16")
            C = T.match_buffer(var_C, (T.int64(1), T.int64(1), vocab_size))
            C_temp = T.alloc_buffer((T.int64(1), T.int64(1), vocab_size), "float16")
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), vocab_size, T.int64(4096)):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(B[v_i0, v_i1, v_k], A[v_k, v_i2])
                    T.writes(C_temp[v_i0, v_i1, v_i2])
                    with T.init():
                        C_temp[v_i0, v_i1, v_i2] = T.float16(0)
                    C_temp[v_i0, v_i1, v_i2] = (
                        C_temp[v_i0, v_i1, v_i2] + B[v_i0, v_i1, v_k] * A[v_k, v_i2]
                    )
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), vocab_size):
                with T.block("epilogue"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(C_temp[v_i0, v_i1, v_i2])
                    T.writes(C[v_i0, v_i1, v_i2])
                    C[v_i0, v_i1, v_i2] = T.Cast("float32", C_temp[v_i0, v_i1, v_i2])

    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def main(var_A: T.handle, B: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_C: T.handle):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            vocab_size = T.int64()
            A = T.match_buffer(var_A, (T.int64(4096), vocab_size), "float16")
            C = T.match_buffer(var_C, (T.int64(1), T.int64(1), vocab_size))
            # with T.block("root"):
            C_temp_local = T.alloc_buffer((T.int64(1), T.int64(1), vocab_size), "float16", scope="local")
            C_temp_rf_local = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(1), vocab_size), "float16", scope="local")
            for ax0_fused_0 in T.thread_binding(vocab_size, thread="blockIdx.x"):
                for ax0_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.x"):
                    for ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        with T.block("matmul_rf_init"):
                            vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                            v0 = T.axis.spatial(vocab_size, ax0_fused_0 + ax0_fused_1)
                            T.reads()
                            T.writes(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = T.float16(0)
                        for ax1_fused_0, u in T.grid(T.int64(256), 1):
                            with T.block("matmul_rf_update"):
                                vax1_fused_1 = T.axis.spatial(T.int64(16), ax1_fused_1)
                                v0 = T.axis.spatial(vocab_size, ax0_fused_0 + ax0_fused_1)
                                vax1_fused_0 = T.axis.reduce(T.int64(256), ax1_fused_0)
                                T.reads(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0], B[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1], A[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0])
                                T.writes(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                                C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] + B[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1] * A[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0]
                for ax1_fused in T.thread_binding(T.int64(1), thread="threadIdx.x"):
                    for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                        with T.block("matmul"):
                            vax1_fused_1, v0 = T.axis.remap("RS", [ax0, ax0_fused_0])
                            T.reads(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            T.writes(C_temp_local[T.int64(0), T.int64(0), v0])
                            with T.init():
                                C_temp_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                            C_temp_local[T.int64(0), T.int64(0), v0] = C_temp_local[T.int64(0), T.int64(0), v0] + C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0]
                for ax0_fused_0_1 in T.thread_binding(T.int64(1), thread="threadIdx.x"):
                    for ax0_fused_1 in range(T.int64(1)):
                        with T.block("epilogue"):
                            v0 = T.axis.spatial(vocab_size, ax0_fused_0)
                            T.reads(C_temp_local[T.int64(0), T.int64(0), v0])
                            T.writes(C[T.int64(0), T.int64(0), v0])
                            C[T.int64(0), T.int64(0), v0] = T.Cast("float32", C_temp_local[T.int64(0), T.int64(0), v0])
    # fmt: on

    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
