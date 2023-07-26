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
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            C_rf_local = T.alloc_buffer((256, 1, 1, 4096), "float16", scope="local")
            for i2_i0_i1_fused in T.thread_binding(4096, thread="blockIdx.x"):
                for k_0_fused_1 in T.thread_binding(256, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("matmul_rf_init"):
                        vk_0_fused_1 = T.axis.spatial(256, k_0_fused_1)
                        v_i2 = T.axis.spatial(4096, i2_i0_i1_fused)
                        C_rf_local[vk_0_fused_1, 0, 0, v_i2] = T.float16(0)
                    for k_0_fused_0, k_1 in T.grid(2, 8):
                        with T.block("matmul_rf_update"):
                            vk_0_fused_1 = T.axis.spatial(256, k_0_fused_1)
                            v_i2, vk_0_fused_0, vk_1 = T.axis.remap("SRR", [i2_i0_i1_fused, k_0_fused_0, k_1])
                            C_rf_local[vk_0_fused_1, 0, 0, v_i2] = C_rf_local[vk_0_fused_1, 0, 0, v_i2] + V[0, 0, vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i2, (vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1) // 8], T.Cast("uint32", (vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i2, (vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1) // 32])
                for ax1_ax2_ax3_fused in range(1): # pylint: disable=unused-variable
                    for ax0_fused in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("matmul"):
                            vk_0_fused_1 = T.axis.reduce(256, ax0_fused)
                            v_i2 = T.axis.spatial(4096, i2_i0_i1_fused)
                            with T.init():
                                C[0, 0, v_i2] = T.float16(0)
                            C[0, 0, v_i2] = C[0, 0, v_i2] + C_rf_local[vk_0_fused_1, 0, 0, v_i2]
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
        def func(W: T.Buffer((512, 4096), "uint32"), S: T.Buffer((128, 4096), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            C_rf_local = T.alloc_buffer((256, 1, 1, 4096), "float16", scope="local")
            for i2_0_i0_i1_fused in T.thread_binding(512, thread="blockIdx.x"):
                for k_fused_1 in T.thread_binding(256, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    for i2_1_init in range(8):
                        with T.block("matmul_rf_init"):
                            vk_fused_1 = T.axis.spatial(256, k_fused_1)
                            v_i2 = T.axis.spatial(4096, i2_0_i0_i1_fused * 8 + i2_1_init)
                            C_rf_local[vk_fused_1, 0, 0, v_i2] = T.float16(0)
                    for k_fused_0, i2_1 in T.grid(16, 8):
                        with T.block("matmul_rf_update"):
                            vk_fused_1 = T.axis.spatial(256, k_fused_1)
                            v_i2 = T.axis.spatial(4096, i2_0_i0_i1_fused * 8 + i2_1)
                            vk_fused_0 = T.axis.reduce(16, k_fused_0)
                            C_rf_local[vk_fused_1, 0, 0, v_i2] = C_rf_local[vk_fused_1, 0, 0, v_i2] + V[0, 0, vk_fused_0 * 256 + vk_fused_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i2 // 8, vk_fused_0 * 256 + vk_fused_1], T.Cast("uint32", v_i2 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i2 // 32, vk_fused_0 * 256 + vk_fused_1])
                for ax1_ax2_ax3_fused_0 in range(1):
                    for ax0_fused in T.thread_binding(256, thread="threadIdx.x"):
                        for ax1_ax2_ax3_fused_1 in range(8):
                            with T.block("matmul"):
                                vk_fused_1 = T.axis.reduce(256, ax0_fused)
                                v_i2 = T.axis.spatial(4096, i2_0_i0_i1_fused * 8 + ax1_ax2_ax3_fused_0 * 8 + ax1_ax2_ax3_fused_1)
                                with T.init():
                                    C[0, 0, v_i2] = T.float16(0)
                                C[0, 0, v_i2] = C[0, 0, v_i2] + C_rf_local[vk_fused_1, 0, 0, v_i2]

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
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), D: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            C_local = T.alloc_buffer((1, 1, 4096), "float16", scope="local")
            C_rf_local = T.alloc_buffer((256, 1, 1, 4096), "float16", scope="local")
            for i2_i0_i1_fused in T.thread_binding(4096, thread="blockIdx.x"):
                for k_0_fused_1 in T.thread_binding(256, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("matmul_rf_init"):
                        vk_0_fused_1 = T.axis.spatial(256, k_0_fused_1)
                        v_i2 = T.axis.spatial(4096, i2_i0_i1_fused)
                        C_rf_local[vk_0_fused_1, 0, 0, v_i2] = T.float16(0)
                    for k_0_fused_0, k_1 in T.grid(2, 8):
                        with T.block("matmul_rf_update"):
                            vk_0_fused_1 = T.axis.spatial(256, k_0_fused_1)
                            v_i2, vk_0_fused_0, vk_1 = T.axis.remap("SRR", [i2_i0_i1_fused, k_0_fused_0, k_1])
                            C_rf_local[vk_0_fused_1, 0, 0, v_i2] = C_rf_local[vk_0_fused_1, 0, 0, v_i2] + V[0, 0, vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i2, (vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1) // 8], T.Cast("uint32", (vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i2, (vk_0_fused_0 * 2048 + vk_0_fused_1 * 8 + vk_1) // 32])
                for ax1_ax2_ax3_fused in range(1):  # pylint: disable=unused-variable
                    for ax0_fused in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("matmul"):
                            vk_0_fused_1 = T.axis.reduce(256, ax0_fused)
                            v_i2 = T.axis.spatial(4096, i2_i0_i1_fused)
                            with T.init():
                                C_local[0, 0, v_i2] = T.float16(0)
                            C_local[0, 0, v_i2] = C_local[0, 0, v_i2] + C_rf_local[vk_0_fused_1, 0, 0, v_i2]
                with T.block("sigmoid"):
                    v_i2 = T.axis.spatial(4096, i2_i0_i1_fused)
                    D[0, 0, v_i2] = T.sigmoid(C_local[0, 0, v_i2])

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
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            C_fp32_local = T.alloc_buffer((1, 1, 4096), scope="local")
            C_fp32_rf_local = T.alloc_buffer((256, 1, 1, 4096), scope="local")
            for ax0_fused in T.thread_binding(4096, thread="blockIdx.x"):
                for ax1_0_fused_1 in T.thread_binding(256, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("matmul_rf_init"):
                        vax1_0_fused_1, v0 = T.axis.remap("SS", [ax1_0_fused_1, ax0_fused])
                        T.reads()
                        T.writes(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                        C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float32(0)
                    for ax1_0_fused_0, ax1_1 in T.grid(2, 8):
                        with T.block("matmul_rf_update"):
                            vax1_0_fused_1, v0, vax1_0_fused_0, vax1_1 = T.axis.remap("SSRR", [ax1_0_fused_1, ax0_fused, ax1_0_fused_0, ax1_1])
                            T.reads(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0], V[0, 0, vax1_0_fused_0 * 2048 + vax1_0_fused_1 * 8 + vax1_1], W[v0, (vax1_0_fused_0 * 2048 + vax1_0_fused_1 * 8 + vax1_1) // 8], S[v0, (vax1_0_fused_0 * 2048 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                            T.writes(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                            C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] = C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] + T.Cast("float32", V[0, 0, vax1_0_fused_0 * 2048 + vax1_0_fused_1 * 8 + vax1_1]) * T.Cast("float32", (T.Cast("float16", T.bitwise_and(T.shift_right(W[v0, (vax1_0_fused_0 * 2048 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 2048 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0, (vax1_0_fused_0 * 2048 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                for ax1_fused in range(1):  # pylint: disable=unused-variable
                    for ax0_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("matmul"):
                            vax1_0_fused_1, v0 = T.axis.remap("RS", [ax0_fused_1, ax0_fused])
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
        def main(A: T.Buffer((1, 1, 4096), "float16"), B: T.Buffer((4096,), "float16"), rms_norm: T.Buffer((1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": True, "tir.is_scheduled": 1})
            # with T.block("root"):
            Ared_temp_shared = T.alloc_buffer((1, 1), scope="shared")
            Ared_temp_rf_local = T.alloc_buffer((256, 1, 1), scope="local")
            for ax0_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"): # pylint: disable=unused-variable
                for ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("Ared_temp_rf_init"):
                        vax1_fused_1 = T.axis.spatial(256, ax1_fused_1)
                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                        T.reads()
                        T.writes(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                        Ared_temp_rf_local[vax1_fused_1, 0, 0] = T.float32(0)
                    for ax1_fused_0, u in T.grid(16, 1): # pylint: disable=unused-variable
                        with T.block("Ared_temp_rf_update"):
                            vax1_fused_1 = T.axis.spatial(256, ax1_fused_1)
                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                            vax1_fused_0 = T.axis.reduce(16, ax1_fused_0)
                            T.reads(Ared_temp_rf_local[vax1_fused_1, 0, 0], A[0, 0, vax1_fused_0 * 256 + vax1_fused_1])
                            T.writes(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                            Ared_temp_rf_local[vax1_fused_1, 0, 0] = Ared_temp_rf_local[vax1_fused_1, 0, 0] + T.Cast("float32", A[0, 0, vax1_fused_0 * 256 + vax1_fused_1]) * T.Cast("float32", A[0, 0, vax1_fused_0 * 256 + vax1_fused_1])
                for ax1_fused in range(T.int64(1)): # pylint: disable=unused-variable
                    for ax0 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("Ared_temp"):
                            vax1_fused_1 = T.axis.reduce(256, ax0)
                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                            T.reads(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                            T.writes(Ared_temp_shared[0, 0])
                            with T.init():
                                Ared_temp_shared[0, 0] = T.float32(0)
                            Ared_temp_shared[0, 0] = Ared_temp_shared[0, 0] + Ared_temp_rf_local[vax1_fused_1, 0, 0]
                for ax0_fused_0 in range(16):
                    for ax0_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("rms_norm"):
                            v0 = T.axis.spatial(4096, ax0_fused_0 * 256 + ax0_fused_1)
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


if __name__ == "__main__":
    tvm.testing.main()
