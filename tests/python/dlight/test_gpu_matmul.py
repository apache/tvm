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
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    @pytest.fixture
    def transform(self):
        def transform(mod):
            with Target("nvidia/geforce-rtx-3090-ti"):
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
        matmul_reindex_pad_local = T.alloc_buffer((T.int64(1), (m + T.int64(15)) // T.int64(16) * T.int64(16), T.int64(4096)), scope="local")
        inp0_reindex_pad_shared = T.alloc_buffer((T.int64(1), (m + T.int64(15)) // T.int64(16) * T.int64(16), T.int64(4096)), scope="shared")
        inp1_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding((m + T.int64(15)) // T.int64(16), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(64), thread="blockIdx.y"):
                    for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                        for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                            for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                    for ax2_3_init, ax1_3_init in T.grid(T.int64(4), T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), ax1_0 * T.int64(16) + ax1_1 * T.int64(16) + ax1_2 * T.int64(2) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_init)
                                            T.reads()
                                            T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0)
                                    for ax3_0 in range(T.int64(256)):
                                        for ax0_ax1_ax2_fused_0 in range(T.int64(1)):
                                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                                for ax0_ax1_ax2_fused_2 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                                    for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                        with T.block("inp0_reindex_pad_shared"):
                                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                            v1 = T.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), ax1_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                            v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                            T.reads(inp0[v0, v1, v2])
                                                            T.writes(inp0_reindex_pad_shared[v0, v1, v2])
                                                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 2]]})
                                                            inp0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
                                        for ax0_ax1_ax2_fused_0 in range(T.int64(4)):
                                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                                for ax0_ax1_ax2_fused_2 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                                    for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                        with T.block("inp1_reindex_shared"):
                                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                            v1 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                            v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                            T.reads(inp1[v2, v1])
                                                            T.writes(inp1_reindex_shared[v0, v1, v2])
                                                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 2]]})
                                                            inp1_reindex_shared[v0, v1, v2] = inp1[v2, v1]
                                        for ax3_1, ax2_3, ax1_3 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), ax1_0 * T.int64(16) + ax1_1 * T.int64(16) + ax1_2 * T.int64(2) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3)
                                                v3 = T.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], inp0_reindex_pad_shared[T.int64(0), v1, v3], inp1_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                                matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + inp0_reindex_pad_shared[T.int64(0), v1, v3] * inp1_reindex_shared[T.int64(0), v2, v3]
                                    for ax0_1, ax1, ax2 in T.grid(T.int64(1), T.int64(2), T.int64(4)):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0_1)
                                            v1 = T.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), ax1_0 * T.int64(16) + ax1_2 * T.int64(2) + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_2 * T.int64(4) + ax2)
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[T.int64(0), v1, v2])
                                            if v1 < m:
                                                matmul[T.int64(0), v1, v2] = matmul_reindex_pad_local[v0, v1, v2]
    # fmt: on


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
        var_matmul_intermediate_reindex_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="local")
        A_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4096)), scope="shared")
        var_decode_intermediate_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
        for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
            for ax1_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(T.int64(64), thread="blockIdx.y"):
                    for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                        for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                            for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                    for ax2_3_init, ax1_3_init in T.grid(T.int64(4), T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(16) + ax1_1 * T.int64(16) + ax1_2 * T.int64(2) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_init)
                                            T.reads()
                                            T.writes(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2])
                                            var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] = T.float32(0)
                                    for ax3_0 in range(T.int64(256)):
                                        for ax0_ax1_ax2_fused_0 in range(T.int64(1)):
                                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                                for ax0_ax1_ax2_fused_2 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                                    for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                        with T.block("A_reindex_shared"):
                                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                            v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                            T.reads(A[v0, v1, v2])
                                                            T.writes(A_reindex_shared[v0, v1, v2])
                                                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 2]]})
                                                            A_reindex_shared[v0, v1, v2] = A[v0, v1, v2]
                                        for ax0_ax1_ax2_fused_0 in range(T.int64(4)):
                                            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                                for ax0_ax1_ax2_fused_2 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                                    for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                        with T.block("var_decode_intermediate_reindex_shared"):
                                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                            v1 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                            v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(256) + ax0_ax1_ax2_fused_1 * T.int64(16) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                            T.reads(W[v2 // T.int64(8), v1], S[v2 // T.int64(32), v1])
                                                            T.writes(var_decode_intermediate_reindex_shared[v0, v1, v2])
                                                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 2]]})
                                                            var_decode_intermediate_reindex_shared[v0, v1, v2] = T.Cast("float32", T.bitwise_and(T.shift_right(W[v2 // T.int64(8), v1], T.Cast("uint32", v2 % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(S[v2 // T.int64(32), v1], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(S[v2 // T.int64(32), v1], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                                        for ax3_1, ax2_3, ax1_3 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), ax0)
                                                v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(16) + ax1_1 * T.int64(16) + ax1_2 * T.int64(2) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3)
                                                v3 = T.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2], A_reindex_shared[T.int64(0), v1, v3], var_decode_intermediate_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2])
                                                var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] = var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] + A_reindex_shared[T.int64(0), v1, v3] * var_decode_intermediate_reindex_shared[T.int64(0), v2, v3]
                                    for ax0_1, ax1, ax2 in T.grid(T.int64(1), T.int64(2), T.int64(4)):
                                        with T.block("var_matmul_intermediate_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0_1)
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(16) + ax1_2 * T.int64(2) + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_2 * T.int64(4) + ax2)
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


if __name__ == "__main__":
    tvm.testing.main()
