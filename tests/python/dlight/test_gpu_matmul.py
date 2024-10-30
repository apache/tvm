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
                return dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)

        return transform


class TestMatmul(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(var_inp0: T.handle, inp1: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_matmul: T.handle):
        m = T.int64()
        inp0 = T.match_buffer(var_inp0, (T.int64(1), m, T.int64(4096)))
        matmul = T.match_buffer(var_matmul, (T.int64(1), m, T.int64(4096)))
        for i0, i1, i2, k in T.grid(T.int64(1), m, T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]

    @T.prim_func
    def expected(var_inp0: T.handle, inp1: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_matmul: T.handle):
        T.func_attr({"tir.is_scheduled": 1})
        m = T.int64()
        inp0 = T.match_buffer(var_inp0, (T.int64(1), m, T.int64(4096)))
        matmul = T.match_buffer(var_matmul, (T.int64(1), m, T.int64(4096)))
        # with T.block("root"):
        matmul_reindex_pad_local = T.alloc_buffer((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="local")
        inp0_reindex_pad_shared = T.alloc_buffer((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="shared")
        inp1_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(64), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding((m + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("inp0_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(inp0[v0, v1, v2])
                                                        T.writes(inp0_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        inp0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("inp1_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(inp1[v2, v1])
                                                        T.writes(inp1_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        inp1_reindex_shared[v0, v1, v2] = inp1[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], inp0_reindex_pad_shared[T.int64(0), v1, v3], inp1_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                                matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + inp0_reindex_pad_shared[T.int64(0), v1, v3] * inp1_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < m)
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[T.int64(0), v1, v2])
                                            matmul[T.int64(0), v1, v2] = matmul_reindex_pad_local[v0, v1, v2]
    # fmt: on


def test_matmul_int32():
    # fmt: off
    @T.prim_func(private=True)
    def func(var_inp0: T.handle, inp1: T.Buffer((4096, 4096), "float32"), var_matmul: T.handle):
        m = T.int32()
        inp0 = T.match_buffer(var_inp0, (1, m, 4096))
        matmul = T.match_buffer(var_matmul, (1, m, 4096))
        for i0, i1, i2, k in T.grid(1, m, 4096, 4096):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]

    @T.prim_func(private=True)
    def expected(var_inp0: T.handle, inp1: T.Buffer((4096, 4096), "float32"), var_matmul: T.handle):
        T.func_attr({"tir.is_scheduled": 1})
        m = T.int32()
        inp0 = T.match_buffer(var_inp0, (1, m, 4096))
        matmul = T.match_buffer(var_matmul, (1, m, 4096))
        # with T.block("root"):
        matmul_reindex_pad_local = T.alloc_buffer((1, (m + 31) // 32 * 32, 4096), scope="local")
        inp0_reindex_pad_shared = T.alloc_buffer((1, (m + 31) // 32 * 32, 4096), scope="shared")
        inp1_reindex_shared = T.alloc_buffer((1, 4096, 4096), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(64, thread="blockIdx.y"):
            for ax1_0 in T.thread_binding((m + 31) // 32, thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(1, thread="vthread.y"):
                    for ax1_1 in T.thread_binding(1, thread="vthread.x"):
                        for ax2_2 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(8, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(4, 2):
                                    for ax2_3_1_init in T.vectorized(2):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3_init)
                                            v2 = T.axis.spatial(4096, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0_init * 2 + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_pad_local[0, v1, v2])
                                            matmul_reindex_pad_local[0, v1, v2] = T.float32(0)
                                for ax3_0 in range(256):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(2):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with T.block("inp0_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = T.axis.spatial(4096, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        T.reads(inp0[v0, v1, v2])
                                                        T.writes(inp0_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        inp0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(4):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with T.block("inp1_reindex_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(4096, ax0_ax2_0_fused * 64 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = T.axis.spatial(4096, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        T.reads(inp1[v2, v1])
                                                        T.writes(inp1_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        inp1_reindex_shared[v0, v1, v2] = inp1[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(16, 4, 2):
                                        for ax2_3_1 in T.vectorized(2):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3)
                                                v2 = T.axis.spatial(4096, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0 * 2 + ax2_3_1)
                                                v3 = T.axis.reduce(4096, ax3_0 * 16 + ax3_1)
                                                T.reads(matmul_reindex_pad_local[0, v1, v2], inp0_reindex_pad_shared[0, v1, v3], inp1_reindex_shared[0, v2, v3])
                                                T.writes(matmul_reindex_pad_local[0, v1, v2])
                                                matmul_reindex_pad_local[0, v1, v2] = matmul_reindex_pad_local[0, v1, v2] + inp0_reindex_pad_shared[0, v1, v3] * inp1_reindex_shared[0, v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(1, 4, 2):
                                    for ax2_1_1 in T.vectorized(2):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_2 * 4 + ax1)
                                            v2 = T.axis.spatial(4096, ax0_ax2_0_fused * 64 + ax2_2 * 4 + ax2_0 * 2 + ax2_1_1)
                                            T.where(ax1_0 * 32 + ax1_2 * 4 + ax1 < m)
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[0, v1, v2])
                                            matmul[0, v1, v2] = matmul_reindex_pad_local[v0, v1, v2]
    # fmt: on

    mod = tvm.IRModule({"main": func})
    with Target("nvidia/geforce-gtx-1080-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


class TestFusedMatmul(BaseBeforeAfter):
    # fmt: off

    @T.prim_func
    def before(W: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), S: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), A: T.Buffer((T.int64(1), T.int64(32), T.int64(4096)), "float32"), C: T.Buffer((T.int64(1), T.int64(32), T.int64(4096)), "float32"), Out: T.Buffer((T.int64(1), T.int64(32), T.int64(4096)), "float32")):
        var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)))
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)))
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(W[v_i // T.int64(8), v_j], S[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(W[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(S[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(S[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(4096)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(C[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(Out[v_ax0, v_ax1, v_ax2])
                Out[v_ax0, v_ax1, v_ax2] = C[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def expected(W: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), S: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), A: T.Buffer((T.int64(1), T.int64(32), T.int64(4096)), "float32"), C: T.Buffer((T.int64(1), T.int64(32), T.int64(4096)), "float32"), Out: T.Buffer((T.int64(1), T.int64(32), T.int64(4096)), "float32")):
        T.func_attr({"tir.is_scheduled": 1})
        # with T.block("root"):
        var_matmul_intermediate_reindex_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="local")
        A_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="shared")
        var_decode_intermediate_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(64), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2])
                                            var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("A_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(A[v0, v1, v2])
                                                        T.writes(A_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        A_reindex_shared[v0, v1, v2] = A[v0, v1, v2]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("var_decode_intermediate_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(W[v2 // T.int64(8), v1], S[v2 // T.int64(32), v1])
                                                        T.writes(var_decode_intermediate_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        var_decode_intermediate_reindex_shared[v0, v1, v2] = T.Cast("float32", T.bitwise_and(T.shift_right(W[v2 // T.int64(8), v1], T.Cast("uint32", v2 % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(S[v2 // T.int64(32), v1], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(S[v2 // T.int64(32), v1], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2], A_reindex_shared[T.int64(0), v1, v3], var_decode_intermediate_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2])
                                                var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] = var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] + A_reindex_shared[T.int64(0), v1, v3] * var_decode_intermediate_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("var_matmul_intermediate_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(C[T.int64(0), v1, v2], var_matmul_intermediate_reindex_local[v0, v1, v2])
                                            T.writes(Out[T.int64(0), v1, v2])
                                            Out[T.int64(0), v1, v2] = C[T.int64(0), v1, v2] + var_matmul_intermediate_reindex_local[v0, v1, v2]

    # fmt: on


class TestSkipGEMV(BaseBeforeAfter):
    # fmt: off

    @T.prim_func
    def before(W: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), S: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), C: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32"), Out: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)))
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)))
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(W[v_i // T.int64(8), v_j], S[v_i // T.int64(32), v_j])
                T.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(W[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(S[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(S[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(C[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(Out[v_ax0, v_ax1, v_ax2])
                Out[v_ax0, v_ax1, v_ax2] = C[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    # fmt: on

    expected = before


class TestOutputFP32(BaseBeforeAfter):
    # fmt: off

    @T.prim_func
    def before(lv13: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv14: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv48: T.handle, lv13_1: T.Buffer((T.int64(4096),), "float16"), p_lv3: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv48 = T.match_buffer(p_lv48, (T.int64(1), n, T.int64(4096)), "float16")
        lv3 = T.match_buffer(p_lv3, (T.int64(1), n, T.int64(4096)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
        var_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
        var_compute_intermediate = T.alloc_buffer((T.int64(4096),))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)))
        var_compute_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv13[v_i, v_j // T.int64(8)], lv14[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv13[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv14[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv48[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv48[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_k, v_i2])
        for i0 in range(T.int64(4096)):
            with T.block("compute"):
                v_i0 = T.axis.spatial(T.int64(4096), i0)
                T.reads(lv13_1[v_i0])
                T.writes(var_compute_intermediate[v_i0])
                var_compute_intermediate[v_i0] = T.Cast("float32", lv13_1[v_i0])
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], var_compute_intermediate[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + var_compute_intermediate[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
                T.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
                var_compute_intermediate_1[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv3[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv3[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def expected(lv13: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv14: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv48: T.handle, lv13_1: T.Buffer((T.int64(4096),), "float16"), p_lv3: T.handle, p_output0: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv48 = T.match_buffer(p_lv48, (T.int64(1), n, T.int64(4096)), "float16")
        lv3 = T.match_buffer(p_lv3, (T.int64(1), n, T.int64(4096)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
        # with T.block("root"):
        var_matmul_intermediate_reindex_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="local")
        lv48_reindex_pad_shared = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), "float16", scope="shared")
        p_output0_intermediate_1_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), "float16", scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(64), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                            var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv48_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv48[v0, v1, v2])
                                                        T.writes(lv48_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv48_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < n, lv48[v0, v1, v2], T.float16(0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("p_output0_intermediate_1_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv13[v2, v1 // T.int64(8)], lv14[v2, v1 // T.int64(32)])
                                                        T.writes(p_output0_intermediate_1_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        p_output0_intermediate_1_reindex_shared[v0, v1, v2] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv13[v2, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv14[v2, v1 // T.int64(32)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2], lv48_reindex_pad_shared[T.int64(0), v1, v3], p_output0_intermediate_1_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                                var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] + T.Cast("float32", lv48_reindex_pad_shared[T.int64(0), v1, v3]) * T.Cast("float32", p_output0_intermediate_1_reindex_shared[T.int64(0), v2, v3])
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("var_matmul_intermediate_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < n)
                                            T.reads(var_matmul_intermediate_reindex_pad_local[v0, v1, v2], lv13_1[v2], lv3[T.int64(0), v1, v2])
                                            T.writes(p_output0_intermediate[T.int64(0), v1, v2])
                                            p_output0_intermediate[T.int64(0), v1, v2] = T.Cast("float16", var_matmul_intermediate_reindex_pad_local[v0, v1, v2] + T.Cast("float32", lv13_1[v2])) + lv3[T.int64(0), v1, v2]

    # fmt: on


class TestInlineConsumerChain(BaseBeforeAfter):
    # fmt: off
    @T.prim_func(private=True)
    def before(p_lv26: T.handle, lv9: T.Buffer((T.int64(2048), T.int64(2048)), "float16"), p_lv52: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv26 = T.match_buffer(p_lv26, (n, T.int64(2048)), "float16")
        lv52 = T.match_buffer(p_lv52, (T.int64(1), n, T.int64(2048)))
        var_T_multiply_intermediate = T.match_buffer(p_output0, (n, T.int64(2048)), "float16")
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((n, T.int64(2048)), "float16")
        compute = T.alloc_buffer((n, T.int64(2048)), "float16")
        var_T_multiply_intermediate_1 = T.alloc_buffer((n, T.int64(2048)), "float16")
        var_T_squeeze_intermediate = T.alloc_buffer((n, T.int64(2048)))
        var_compute_intermediate = T.alloc_buffer((n, T.int64(2048)), "float16")
        for i0, i1, k in T.grid(n, T.int64(2048), T.int64(2048)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv26[v_i0, v_k], lv9[v_i1, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1] = var_NT_matmul_intermediate[v_i0, v_i1] + lv26[v_i0, v_k] * lv9[v_i1, v_k]
        for i0, i1 in T.grid(n, T.int64(2048)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(var_NT_matmul_intermediate[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(var_NT_matmul_intermediate[v_i0, v_i1])
        for ax0, ax1 in T.grid(n, T.int64(2048)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1])
                var_T_multiply_intermediate_1[v_ax0, v_ax1] = var_NT_matmul_intermediate[v_ax0, v_ax1] * compute[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(n, T.int64(2048)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv52[T.int64(0), v_ax0, v_ax1])
                T.writes(var_T_squeeze_intermediate[v_ax0, v_ax1])
                var_T_squeeze_intermediate[v_ax0, v_ax1] = lv52[T.int64(0), v_ax0, v_ax1]
        for i0, i1 in T.grid(n, T.int64(2048)):
            with T.block("compute_1"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(var_T_squeeze_intermediate[v_i0, v_i1])
                T.writes(var_compute_intermediate[v_i0, v_i1])
                var_compute_intermediate[v_i0, v_i1] = T.Cast("float16", var_T_squeeze_intermediate[v_i0, v_i1])
        for ax0, ax1 in T.grid(n, T.int64(2048)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(var_compute_intermediate[v_ax0, v_ax1], var_T_multiply_intermediate_1[v_ax0, v_ax1])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1])
                var_T_multiply_intermediate[v_ax0, v_ax1] = var_compute_intermediate[v_ax0, v_ax1] * var_T_multiply_intermediate_1[v_ax0, v_ax1]

    @T.prim_func
    def expected(p_lv26: T.handle, lv9: T.Buffer((T.int64(2048), T.int64(2048)), "float16"), p_lv52: T.handle, p_output0: T.handle):
        T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        n = T.int64()
        lv26 = T.match_buffer(p_lv26, (n, T.int64(2048)), "float16")
        lv52 = T.match_buffer(p_lv52, (T.int64(1), n, T.int64(2048)))
        var_T_multiply_intermediate = T.match_buffer(p_output0, (n, T.int64(2048)), "float16")
        # with T.block("root"):
        var_NT_matmul_intermediate_reindex_pad_local = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2048)), "float16", scope="local")
        lv26_reindex_pad_shared = T.alloc_buffer((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2048)), "float16", scope="shared")
        lv9_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(2048)), "float16", scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(32), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding((n + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("NT_matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                            var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = T.float16(0)
                                for ax3_0 in range(T.int64(128)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv26_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv26[v1, v2])
                                                        T.writes(lv26_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv26_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < n, lv26[v1, v2], T.float16(0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv9_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv9[v1, v2])
                                                        T.writes(lv9_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv9_reindex_shared[v0, v1, v2] = lv9[v1, v2]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("NT_matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(2048), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2], lv26_reindex_pad_shared[T.int64(0), v1, v3], lv9_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                                var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] + lv26_reindex_pad_shared[T.int64(0), v1, v3] * lv9_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("var_NT_matmul_intermediate_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(lv52[T.int64(0), v1, v2], var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2])
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < n)
                                            T.writes(var_T_multiply_intermediate[v1, v2])
                                            var_T_multiply_intermediate[v1, v2] = T.Cast("float16", lv52[T.int64(0), v1, v2]) * (var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2] * T.sigmoid(var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2]))

    # fmt: on


class AndroidBeforeAfter(tvm.testing.CompareBeforeAfter):
    @pytest.fixture
    def transform(self):
        def transform(mod):
            with Target("opencl", host="llvm -mtriple=aarch64-linux-android"):
                return dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)

        return transform


class TestMatmulAndroid(AndroidBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(var_inp0: T.handle, inp1: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_matmul: T.handle):
        m = T.int64()
        inp0 = T.match_buffer(var_inp0, (T.int64(1), m, T.int64(4096)))
        matmul = T.match_buffer(var_matmul, (T.int64(1), m, T.int64(4096)))
        for i0, i1, i2, k in T.grid(T.int64(1), m, T.int64(4096), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]

    @T.prim_func
    def expected(var_inp0: T.handle, inp1: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_matmul: T.handle):
        T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1})
        m = T.int64()
        inp0 = T.match_buffer(var_inp0, (T.int64(1), m, T.int64(4096)))
        matmul = T.match_buffer(var_matmul, (T.int64(1), m, T.int64(4096)))
        # with T.block("root"):
        inp0_reindex_pad = T.alloc_buffer((T.int64(1), (m + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)))
        matmul_pad_local = T.alloc_buffer((T.int64(1), (m + T.int64(15)) // T.int64(16) * T.int64(16), T.int64(4096)), scope="local")
        inp0_reindex_pad_local = T.alloc_buffer((T.int64(1), (m + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)), scope="local")
        for i0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for i1_0 in T.thread_binding(((m + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), thread="blockIdx.y"):
                for i2_0 in T.thread_binding(T.int64(128), thread="blockIdx.x"):
                    for i1_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for i1_2 in T.vectorized(T.int64(16)):
                                with T.block("inp0_reindex_pad"):
                                    v0 = T.axis.spatial(T.int64(1), i0)
                                    v1 = T.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), i1_0 * T.int64(64) + i1_1 * T.int64(16) + i1_2)
                                    v2 = T.axis.spatial(T.int64(4096), i2_0 * T.int64(32) + i2_1)
                                    T.where((i1_0 * T.int64(4) + i1_1) * T.int64(16) + i1_2 < (m + T.int64(15)) // T.int64(16) * T.int64(16))
                                    T.reads(inp0[v0, v1, v2])
                                    T.writes(inp0_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)])
                                    inp0_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
        for i2_0 in T.thread_binding(T.int64(16), thread="blockIdx.x"):
            for i0_i1_fused_0 in T.thread_binding(((m + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), thread="blockIdx.y"):
                for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for i0_i1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for i0_i1_fused_2_init in T.unroll(T.int64(16)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2_init)
                                    v_i2 = T.axis.spatial(T.int64(4096), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2_init)
                                    T.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2_init < (m + T.int64(15)) // T.int64(16) * T.int64(16))
                                    T.reads()
                                    T.writes(matmul_pad_local[v_i0, v_i1, v_i2])
                                    matmul_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                        for k_0, k_1 in T.grid(T.int64(128), T.int64(4)):
                            for k_2 in T.unroll(T.int64(8)):
                                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                    for ax3 in T.vectorized(T.int64(16)):
                                        with T.block("inp0_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial((m + T.int64(15)) // T.int64(16), i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                            v3 = T.axis.spatial(T.int64(16), ax3)
                                            T.where(i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 < (m + T.int64(15)) // T.int64(16))
                                            T.reads(inp0_reindex_pad[v0, v1, v2, v3])
                                            T.writes(inp0_reindex_pad_local[v0, v1, v2, v3])
                                            inp0_reindex_pad_local[v0, v1, v2, v3] = inp0_reindex_pad[v0, v1, v2, v3]
                                for i0_i1_fused_2 in T.unroll(T.int64(16)):
                                    for i2_2 in T.vectorized(T.int64(8)):
                                        with T.block("matmul_update"):
                                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v_i1 = T.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2)
                                            v_i2 = T.axis.spatial(T.int64(4096), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2)
                                            v_k = T.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                            T.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2 < (m + T.int64(15)) // T.int64(16) * T.int64(16))
                                            T.reads(matmul_pad_local[v_i0, v_i1, v_i2], inp0_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)], inp1[v_k, v_i2])
                                            T.writes(matmul_pad_local[v_i0, v_i1, v_i2])
                                            matmul_pad_local[v_i0, v_i1, v_i2] = matmul_pad_local[v_i0, v_i1, v_i2] + inp0_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)] * inp1[v_k, v_i2]
                        for ax0 in T.unroll(T.int64(16)):
                            for ax1 in T.vectorized(T.int64(8)):
                                with T.block("matmul_pad"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(m, i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0)
                                    v2 = T.axis.spatial(T.int64(4096), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                    T.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 - (m + T.int64(15)) // T.int64(16) < T.int64(0) or i0_i1_fused_0 == T.int64(0)) and i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0 < m)
                                    T.reads(matmul_pad_local[v0, v1, v2])
                                    T.writes(matmul[v0, v1, v2])
                                    matmul[v0, v1, v2] = matmul_pad_local[v0, v1, v2]


class TestFusedDequantMatmulAndroid(AndroidBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(lv452: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), lv453: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
        matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(12288)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv452[v_i0 // T.int64(8), v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv452[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with T.block("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], lv453[v_i0 // T.int64(32), v_i1])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * lv453[v_i0 // T.int64(32), v_i1]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_k, v_i2])
                T.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]

    @T.prim_func
    def expected(lv452: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), lv453: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_rms_norm130: T.handle, transformer_h_0_attn_c_attn_bias3: T.Buffer((T.int64(12288),), "float16"), p_output0: T.handle):
        T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        rms_norm130 = T.match_buffer(p_rms_norm130, (T.int64(1), seq_len, T.int64(4096)), "float16")
        T_add_intermediate_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(12288)), "float16")
        # with T.block("root"):
        dequantize_intermediate_intermediate_local = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16", scope="local")
        rms_norm130_reindex_pad = T.alloc_buffer((T.int64(1), (seq_len + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)), "float16")
        matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (seq_len + T.int64(15)) // T.int64(16) * T.int64(16), T.int64(12288)), "float16", scope="local")
        rms_norm130_reindex_pad_local = T.alloc_buffer((T.int64(1), (seq_len + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)), "float16", scope="local")
        lv452_local = T.alloc_buffer((T.int64(512), T.int64(12288)), "uint32", scope="local")
        lv453_local = T.alloc_buffer((T.int64(128), T.int64(12288)), "float16", scope="local")
        for i0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for i1_0 in T.thread_binding(((seq_len + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), thread="blockIdx.y"):
                for i2_0 in T.thread_binding(T.int64(128), thread="blockIdx.x"):
                    for i1_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for i1_2 in T.vectorized(T.int64(16)):
                                with T.block("rms_norm130_reindex_pad"):
                                    v0 = T.axis.spatial(T.int64(1), i0)
                                    v1 = T.axis.spatial((seq_len + T.int64(15)) // T.int64(16) * T.int64(16), i1_0 * T.int64(64) + i1_1 * T.int64(16) + i1_2)
                                    v2 = T.axis.spatial(T.int64(4096), i2_0 * T.int64(32) + i2_1)
                                    T.where((i1_0 * T.int64(4) + i1_1) * T.int64(16) + i1_2 < (seq_len + T.int64(15)) // T.int64(16) * T.int64(16))
                                    T.reads(rms_norm130[v0, v1, v2])
                                    T.writes(rms_norm130_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)])
                                    rms_norm130_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)] = T.if_then_else(v1 < seq_len, rms_norm130[v0, v1, v2], T.float16(0))
        for i2_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for i0_i1_fused_0 in T.thread_binding(((seq_len + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), thread="blockIdx.y"):
                for i2_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for i0_i1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for i0_i1_fused_2_init in T.unroll(T.int64(16)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with T.block("matmul_init"):
                                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = T.axis.spatial((seq_len + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2_init)
                                    v_i2 = T.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2_init)
                                    T.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2_init < (seq_len + T.int64(15)) // T.int64(16) * T.int64(16))
                                    T.reads()
                                    T.writes(matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                    matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_0 in range(T.int64(128)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with T.block("lv453_local"):
                                        v0 = T.axis.spatial(T.int64(128), k_0 + ax0)
                                        v1 = T.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                        T.reads(lv453[v0, v1])
                                        T.writes(lv453_local[v0, v1])
                                        lv453_local[v0, v1] = lv453[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with T.block("lv452_local"):
                                            v0 = T.axis.spatial(T.int64(512), k_0 * T.int64(4) + k_1 + ax0)
                                            v1 = T.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                            T.reads(lv452[v0, v1])
                                            T.writes(lv452_local[v0, v1])
                                            lv452_local[v0, v1] = lv452[v0, v1]
                                for k_2 in T.unroll(T.int64(8)):
                                    for ax0 in T.vectorized(T.int64(8)):
                                        with T.block("dequantize"):
                                            v_i0 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                            v_i1 = T.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax0)
                                            T.reads(lv452_local[v_i0 // T.int64(8), v_i1], lv453_local[v_i0 // T.int64(32), v_i1])
                                            T.writes(dequantize_intermediate_intermediate_local[v_i0, v_i1])
                                            dequantize_intermediate_intermediate_local[v_i0, v_i1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv452_local[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7)) * lv453_local[v_i0 // T.int64(32), v_i1]
                                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                        for ax3 in T.vectorized(T.int64(16)):
                                            with T.block("rms_norm130_reindex_pad_local"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((seq_len + T.int64(15)) // T.int64(16), i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 + ax1)
                                                v2 = T.axis.spatial(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                                v3 = T.axis.spatial(T.int64(16), ax3)
                                                T.where(i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 < (seq_len + T.int64(15)) // T.int64(16))
                                                T.reads(rms_norm130_reindex_pad[v0, v1, v2, v3])
                                                T.writes(rms_norm130_reindex_pad_local[v0, v1, v2, v3])
                                                rms_norm130_reindex_pad_local[v0, v1, v2, v3] = rms_norm130_reindex_pad[v0, v1, v2, v3]
                                    for i0_i1_fused_2 in T.unroll(T.int64(16)):
                                        for i2_2 in T.vectorized(T.int64(8)):
                                            with T.block("matmul_update"):
                                                v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v_i1 = T.axis.spatial((seq_len + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2)
                                                v_i2 = T.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2)
                                                v_k = T.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                                T.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2 < (seq_len + T.int64(15)) // T.int64(16) * T.int64(16))
                                                T.reads(matmul_intermediate_pad_local[v_i0, v_i1, v_i2], rms_norm130_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)], dequantize_intermediate_intermediate_local[v_k, v_i2])
                                                T.writes(matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                                matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + rms_norm130_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)] * dequantize_intermediate_intermediate_local[v_k, v_i2]
                        for ax0 in T.unroll(T.int64(16)):
                            for ax1 in T.vectorized(T.int64(8)):
                                with T.block("T_add"):
                                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v_ax1 = T.axis.spatial(seq_len, i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0)
                                    v_ax2 = T.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                    T.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 - (seq_len + T.int64(15)) // T.int64(16) < T.int64(0) or i0_i1_fused_0 == T.int64(0)) and i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0 < seq_len)
                                    T.reads(matmul_intermediate_pad_local[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                                    T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                                    T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate_pad_local[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]
    # fmt: on


if __name__ == "__main__":
    tvm.testing.main()
