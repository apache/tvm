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
import tvm
import tvm.testing
from tvm import IRModule
from tvm.script import tir as T
from tvm.tir import PrimFunc


def _check_primfunc_transform(before: PrimFunc, expected: PrimFunc):
    before_module = IRModule.from_expr(before)
    after_module = tvm.tir.transform.InjectPermutedLayout()(before_module)

    after = after_module["before"].without_attr("global_symbol")
    expected = expected.without_attr("global_symbol")

    tvm.ir.assert_structural_equal(after, expected)


# This pass is adapted from another previous pass, so we need to ensure backward compatibility here
def test_backward_compatibility_shared_a():
    # fmt: off
    @T.prim_func
    def before(X: T.Buffer((4096, 4096), "float16")):
        # with T.block("root"):
        for blockIdx_y in T.thread_binding(256, thread="blockIdx.y"):
            for threadIdx_y in T.thread_binding(4, thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(32, thread="threadIdx.x"):
                    with T.block(""):
                        T.reads(X[blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4:blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4 + 97, threadIdx_x % 4 * 8:threadIdx_x % 4 * 8 + 4072])
                        T.writes()
                        for ax2_0_0 in range(128):
                            with T.block(""):
                                T.reads(X[blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4:blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4 + 97, ax2_0_0 * 32 + threadIdx_x % 4 * 8:ax2_0_0 * 32 + threadIdx_x % 4 * 8 + 8])
                                T.writes()
                                X_reindex_shared_dyn = T.alloc_buffer((128, 32), "float16", strides=(32, 1), scope="shared.dyn")
                                with T.block("X_reindex_shared.dyn"):
                                    T.reads(X[blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4:blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4 + 97, ax2_0_0 * 32 + threadIdx_x % 4 * 8:ax2_0_0 * 32 + threadIdx_x % 4 * 8 + 8])
                                    T.writes(X_reindex_shared_dyn[threadIdx_y * 8 + threadIdx_x // 4:threadIdx_y * 8 + threadIdx_x // 4 + 97, threadIdx_x % 4 * 8:threadIdx_x % 4 * 8 + 8])
                                    T.block_attr({"permuted_layout": "g2s_A"})
                                    for ax0_ax1_fused_0 in range(4):
                                        for ax0_ax1_fused_3 in T.vectorized(8):
                                            X_reindex_shared_dyn[ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, threadIdx_x % 4 * 8 + ax0_ax1_fused_3] = X[blockIdx_y // 8 * 128 + ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, ax2_0_0 * 32 + threadIdx_x % 4 * 8 + ax0_ax1_fused_3]
                                for ax2_0_1 in range(4):
                                    with T.block(""):
                                        T.reads(X_reindex_shared_dyn[threadIdx_y // 2 * 64:threadIdx_y // 2 * 64 + 64, ax2_0_1 * 8:ax2_0_1 * 8 + 8])
                                        T.writes()
                                        X_reindex_shared_dyn_m16n8k8_matrixA = T.alloc_buffer((64, 8), "float16", scope="m16n8k8.matrixA")
                                        for ax0_0, ax1_0 in T.grid(2, 1):
                                            with T.block("X_reindex_shared.dyn_m16n8k8.matrixA_o"):
                                                T.reads(X_reindex_shared_dyn[threadIdx_y // 2 * 64 + ax0_0 * 32:threadIdx_y // 2 * 64 + ax0_0 * 32 + 32, ax2_0_1 * 8:ax2_0_1 * 8 + 8])
                                                T.writes(X_reindex_shared_dyn_m16n8k8_matrixA[ax0_0 * 32:ax0_0 * 32 + 32, 0:8])
                                                T.block_attr({"permuted_layout": "s2l_A"})
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", X_reindex_shared_dyn_m16n8k8_matrixA.data, ax0_0 * 8, T.tvm_access_ptr(T.type_annotation("float16"), X_reindex_shared_dyn.data, threadIdx_y // 2 * 2048 + ax0_0 * 1024 + ax2_0_1 * 8, 1024, 1), threadIdx_x * 32)

    @T.prim_func
    def expected(X: T.Buffer((4096, 4096), "float16")):
        for blockIdx_y in T.thread_binding(256, thread="blockIdx.y"):
            for threadIdx_y in T.thread_binding(4, thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(32, thread="threadIdx.x"):
                    with T.block(""):
                        for ax2_0_0 in T.serial(128):
                            with T.block(""):
                                X_reindex_shared_dyn = T.alloc_buffer((128, 32), "float16", strides=(32, 1), scope="shared.dyn")
                                with T.block("X_reindex_shared.dyn"):
                                    # annotate the reads and writes because they cannot be inferred from tir.bitwise_xor
                                    T.reads(X[blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4:blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4 + 97, ax2_0_0 * 32 + threadIdx_x % 4 * 8:ax2_0_0 * 32 + threadIdx_x % 4 * 8 + 8])
                                    T.writes(X_reindex_shared_dyn[threadIdx_y * 8 + threadIdx_x // 4:threadIdx_y * 8 + threadIdx_x // 4 + 97, threadIdx_x % 4 * 8:threadIdx_x % 4 * 8 + 8])
                                    for ax0_ax1_fused_0 in range(4):
                                        for ax0_ax1_fused_3 in T.vectorized(8):
                                            X_reindex_shared_dyn[ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, T.bitwise_xor(threadIdx_x % 4, threadIdx_x // 8) * 8 + ax0_ax1_fused_3] = X[blockIdx_y // 8 * 128 + ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, ax2_0_0 * 32 + threadIdx_x % 4 * 8 + ax0_ax1_fused_3]
                                for ax2_0_1 in T.serial(4):
                                    with T.block(""):
                                        X_reindex_shared_dyn_m16n8k8_matrixA = T.alloc_buffer((64, 8), "float16", scope="m16n8k8.matrixA")
                                        for ax0_0, ax1_0 in T.grid(2, 1):
                                            with T.block("X_reindex_shared.dyn_m16n8k8.matrixA_o"):
                                                T.reads(X_reindex_shared_dyn[threadIdx_y // 2 * 64 + ax0_0 * 32:threadIdx_y // 2 * 64 + ax0_0 * 32 + 32, ax2_0_1 * 8:ax2_0_1 * 8 + 8])
                                                T.writes(X_reindex_shared_dyn_m16n8k8_matrixA[ax0_0 * 32:ax0_0 * 32 + 32, 0:8])
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", X_reindex_shared_dyn_m16n8k8_matrixA.data, ax0_0 * 8, T.tvm_access_ptr(T.type_annotation("float16"), X_reindex_shared_dyn.data, threadIdx_y // 2 * 2048 + ax0_0 * 1024 + threadIdx_x * 32 + T.bitwise_xor(ax2_0_1, threadIdx_x % 8 // 2) * 8, 1024, 1), 0)
    # fmt: on
    _check_primfunc_transform(before, expected)


def test_backward_compatibility_shared_a_and_b():
    # fmt: off
    @T.prim_func
    def before(X: T.Buffer((4096, 4096), "float16"), Y: T.Buffer((4096, 4096), "float16")):
        for blockIdx_x in T.thread_binding(4, thread="blockIdx.x"):
            for blockIdx_y in T.thread_binding(256, thread="blockIdx.y"):
                for threadIdx_y in T.thread_binding(4, thread="threadIdx.y"):
                    for threadIdx_x in T.thread_binding(32, thread="threadIdx.x"):
                        with T.block(""):
                            for ax2_0_0 in T.serial(128):
                                with T.block(""):
                                    X_reindex_shared_dyn = T.alloc_buffer((128, 32), "float16", strides=(32, 1), scope="shared.dyn")
                                    Y_reindex_shared_dyn = T.alloc_buffer((32, 128), "float16", strides=(128, 1), scope="shared.dyn")
                                    with T.block("X_reindex_shared.dyn"):
                                        T.block_attr({"permuted_layout": "g2s_A"})
                                        for ax0_ax1_fused_0 in range(4):
                                            for ax0_ax1_fused_3 in T.vectorized(8):
                                                X_reindex_shared_dyn[ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, threadIdx_x % 4 * 8 + ax0_ax1_fused_3] = X[blockIdx_y // 8 * 128 + ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, ax2_0_0 * 32 + threadIdx_x % 4 * 8 + ax0_ax1_fused_3]
                                    with T.block("Y_reindex_shared.dyn"):
                                        T.block_attr({"permuted_layout": "g2s_B"})
                                        for ax0_ax1_fused_0 in range(4):
                                            for ax0_ax1_fused_3 in T.vectorized(8):
                                                Y_reindex_shared_dyn[ax0_ax1_fused_0 * 8 + threadIdx_y * 2 + threadIdx_x // 16, threadIdx_x % 16 * 8 + ax0_ax1_fused_3] = Y[ax2_0_0 * 32 + ax0_ax1_fused_0 * 8 + threadIdx_y * 2 + threadIdx_x // 16, blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8 + ax0_ax1_fused_3]
                                    for ax2_0_1 in T.serial(4):
                                        with T.block(""):
                                            X_reindex_shared_dyn_m16n8k8_matrixA = T.alloc_buffer((64, 8), "float16", scope="m16n8k8.matrixA")
                                            Y_reindex_shared_dyn_m16n8k8_matrixB = T.alloc_buffer((8, 64), "float16", scope="m16n8k8.matrixB")
                                            for ax0_0, ax1_0 in T.grid(2, 1):
                                                with T.block("X_reindex_shared.dyn_m16n8k8.matrixA_o"):
                                                    T.reads(X_reindex_shared_dyn[threadIdx_y // 2 * 64 + ax0_0 * 32:threadIdx_y // 2 * 64 + ax0_0 * 32 + 32, ax2_0_1 * 8:ax2_0_1 * 8 + 8])
                                                    T.writes(X_reindex_shared_dyn_m16n8k8_matrixA[ax0_0 * 32:ax0_0 * 32 + 32, 0:8])
                                                    T.block_attr({"permuted_layout": "s2l_A"})
                                                    T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", X_reindex_shared_dyn_m16n8k8_matrixA.data, ax0_0 * 8, T.tvm_access_ptr(T.type_annotation("float16"), X_reindex_shared_dyn.data, threadIdx_y // 2 * 2048 + ax0_0 * 1024 + ax2_0_1 * 8, 1024, 1), threadIdx_x * 32)
                                            for ax0_0, ax1_0 in T.grid(1, 2):
                                                with T.block("Y_reindex_shared.dyn_m16n8k8.matrixB_o"):
                                                    T.reads(Y_reindex_shared_dyn[ax2_0_1 * 8:ax2_0_1 * 8 + 8, threadIdx_y % 2 * 64 + ax1_0 * 32:threadIdx_y % 2 * 64 + ax1_0 * 32 + 32])
                                                    T.writes(Y_reindex_shared_dyn_m16n8k8_matrixB[0:8, ax1_0 * 32:ax1_0 * 32 + 32])
                                                    T.block_attr({"permuted_layout": "s2l_B"})
                                                    T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", Y_reindex_shared_dyn_m16n8k8_matrixB.data, ax1_0 * 8, T.tvm_access_ptr(T.type_annotation("float16"), Y_reindex_shared_dyn.data, ax2_0_1 * 1024 + threadIdx_y % 2 * 64 + ax1_0 * 32, 1024, 1), threadIdx_x % 8 * 128 + threadIdx_x // 8 * 8)

    @T.prim_func
    def expected(X: T.Buffer((4096, 4096), "float16"), Y: T.Buffer((4096, 4096), "float16")):
        for blockIdx_x in T.thread_binding(4, thread="blockIdx.x"):
            for blockIdx_y in T.thread_binding(256, thread="blockIdx.y"):
                for threadIdx_y in T.thread_binding(4, thread="threadIdx.y"):
                    for threadIdx_x in T.thread_binding(32, thread="threadIdx.x"):
                        with T.block(""):
                            T.reads(X[blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4:blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4 + 97, threadIdx_x % 4 * 8:threadIdx_x % 4 * 8 + 4072], Y[threadIdx_y * 2 + threadIdx_x // 16:threadIdx_y * 2 + threadIdx_x // 16 + 4089, blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8:blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8 + 8])
                            T.writes()
                            for ax2_0_0 in T.serial(128):
                                with T.block(""):
                                    T.reads(X[blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4:blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4 + 97, ax2_0_0 * 32 + threadIdx_x % 4 * 8:ax2_0_0 * 32 + threadIdx_x % 4 * 8 + 8], Y[ax2_0_0 * 32 + threadIdx_y * 2 + threadIdx_x // 16:ax2_0_0 * 32 + threadIdx_y * 2 + threadIdx_x // 16 + 25, blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8:blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8 + 8])
                                    T.writes()
                                    X_reindex_shared_dyn = T.alloc_buffer((128, 32), "float16", strides=(32, 1), scope="shared.dyn")
                                    Y_reindex_shared_dyn = T.alloc_buffer((32, 128), "float16", strides=(128, 1), scope="shared.dyn")
                                    with T.block("X_reindex_shared.dyn"):
                                        T.reads(X[blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4:blockIdx_y // 8 * 128 + threadIdx_y * 8 + threadIdx_x // 4 + 97, ax2_0_0 * 32 + threadIdx_x % 4 * 8:ax2_0_0 * 32 + threadIdx_x % 4 * 8 + 8])
                                        T.writes(X_reindex_shared_dyn[threadIdx_y * 8 + threadIdx_x // 4:threadIdx_y * 8 + threadIdx_x // 4 + 97, threadIdx_x % 4 * 8:threadIdx_x % 4 * 8 + 8])
                                        for ax0_ax1_fused_0 in range(4):
                                            for ax0_ax1_fused_3 in T.vectorized(8):
                                                X_reindex_shared_dyn[ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, T.bitwise_xor(threadIdx_x % 4, threadIdx_x // 8) * 8 + ax0_ax1_fused_3] = X[blockIdx_y // 8 * 128 + ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, ax2_0_0 * 32 + threadIdx_x % 4 * 8 + ax0_ax1_fused_3]
                                    with T.block("Y_reindex_shared.dyn"):
                                        T.reads(Y[ax2_0_0 * 32 + threadIdx_y * 2 + threadIdx_x // 16:ax2_0_0 * 32 + threadIdx_y * 2 + threadIdx_x // 16 + 25, blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8:blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8 + 8])
                                        T.writes(Y_reindex_shared_dyn[threadIdx_y * 2 + threadIdx_x // 16:threadIdx_y * 2 + threadIdx_x // 16 + 25, threadIdx_x % 16 * 8:threadIdx_x % 16 * 8 + 8])
                                        for ax0_ax1_fused_0 in range(4):
                                            for ax0_ax1_fused_3 in T.vectorized(8):
                                                Y_reindex_shared_dyn[ax0_ax1_fused_0 * 8 + threadIdx_y * 2 + threadIdx_x // 16, T.bitwise_xor(threadIdx_x % 16, threadIdx_y * 2 + threadIdx_x // 16) * 8 + ax0_ax1_fused_3]   = Y[ax2_0_0 * 32 + ax0_ax1_fused_0 * 8 + threadIdx_y * 2 + threadIdx_x // 16, blockIdx_x * 1024 + blockIdx_y % 8 * 128 + threadIdx_x % 16 * 8 + ax0_ax1_fused_3]
                                    for ax2_0_1 in T.serial(4):
                                        with T.block(""):
                                            X_reindex_shared_dyn_m16n8k8_matrixA = T.alloc_buffer((64, 8), "float16", scope="m16n8k8.matrixA")
                                            Y_reindex_shared_dyn_m16n8k8_matrixB = T.alloc_buffer((8, 64), "float16", scope="m16n8k8.matrixB")
                                            for ax0_0, ax1_0 in T.grid(2, 1):
                                                with T.block("X_reindex_shared.dyn_m16n8k8.matrixA_o"):
                                                    T.reads(X_reindex_shared_dyn[threadIdx_y // 2 * 64 + ax0_0 * 32:threadIdx_y // 2 * 64 + ax0_0 * 32 + 32, ax2_0_1 * 8:ax2_0_1 * 8 + 8])
                                                    T.writes(X_reindex_shared_dyn_m16n8k8_matrixA[ax0_0 * 32:ax0_0 * 32 + 32, 0:8])
                                                    T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", X_reindex_shared_dyn_m16n8k8_matrixA.data, ax0_0 * 8, T.tvm_access_ptr(T.type_annotation("float16"), X_reindex_shared_dyn.data, threadIdx_y // 2 * 2048 + ax0_0 * 1024 + threadIdx_x * 32 + T.bitwise_xor(ax2_0_1, threadIdx_x % 8 // 2) * 8, 1024, 1), 0)
                                            for ax0_0, ax1_0 in T.grid(1, 2):
                                                with T.block("Y_reindex_shared.dyn_m16n8k8.matrixB_o"):
                                                    T.reads(Y_reindex_shared_dyn[ax2_0_1 * 8:ax2_0_1 * 8 + 8, threadIdx_y % 2 * 64 + ax1_0 * 32:threadIdx_y % 2 * 64 + ax1_0 * 32 + 32])
                                                    T.writes(Y_reindex_shared_dyn_m16n8k8_matrixB[0:8, ax1_0 * 32:ax1_0 * 32 + 32])
                                                    T.ptx_ldmatrix("float16", T.bool(True), 4, ".b16", Y_reindex_shared_dyn_m16n8k8_matrixB.data, ax1_0 * 8, T.tvm_access_ptr(T.type_annotation("float16"), Y_reindex_shared_dyn.data, ax2_0_1 * 1024 + threadIdx_x % 8 * 128 + T.bitwise_xor(threadIdx_y % 2 * 8 + ax1_0 * 4 + threadIdx_x // 8, threadIdx_x % 8) * 8, 1024, 1), 0)
    # fmt: on
    _check_primfunc_transform(before, expected)


def test_buffer_a():
    # fmt: off
    @T.prim_func
    def before(p_A: T.handle):
        A = T.match_buffer(p_A, (T.int64(128), T.int64(32)), "float16")
        A_shared_dyn = T.alloc_buffer((T.int64(128), T.int64(32)), "float16", scope="shared.dyn")
        A_warp = T.alloc_buffer((T.int64(4), T.int64(1), T.int64(32), T.int64(8)), "float16", scope="warp")
        for threadIdx_z in T.thread_binding(T.int64(2), thread="threadIdx.z"):
            for threadIdx_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for v0 in range(T.int64(4)):
                        for v1 in T.vectorized(T.int64(8)):
                            with T.block("A_reindex_shared.dyn"):
                                T.block_attr({"permuted_layout": 1})
                                A_shared_dyn[
                                    v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4),
                                    threadIdx_x % T.int64(4) * T.int64(8) + v1
                                ] = A[
                                    (v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4)) % T.int64(32),
                                    threadIdx_x % T.int64(4) * T.int64(8) + v1
                                ]
                    for v0, v1 in T.grid(T.int64(2), T.int64(4)):
                        with T.block("A_reindex_shared.dyn_warp_o"):
                            T.block_attr({"permuted_layout": 1})
                            with T.block("A_reindex_shared.dyn_warp_o"):
                                T.reads(A_shared_dyn[threadIdx_z * T.int64(64) + v1 * T.int64(16):threadIdx_z * T.int64(64) + v1 * T.int64(16) + T.int64(16), v0 * T.int64(16):v0 * T.int64(16) + T.int64(16)])
                                T.writes(A_warp[v1, T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16",
                                    A_warp.data,
                                    v1 * T.int64(256) + threadIdx_x * T.int64(8),
                                    T.tvm_access_ptr(T.type_annotation("float16"),
                                        A_shared_dyn.data,
                                        threadIdx_z * T.int64(2048) + v1 * T.int64(512) + v0 * T.int64(16), T.int64(512),
                                        1
                                    ),
                                    threadIdx_x % T.int64(16) * T.int64(32) + threadIdx_x // T.int64(16) * T.int64(8)
                                )

    @T.prim_func
    def expected(A: T.Buffer((T.int64(128), T.int64(32)), "float16")):
        A_shared_dyn = T.alloc_buffer((T.int64(128), T.int64(32)), "float16", scope="shared.dyn")
        A_warp = T.alloc_buffer((T.int64(4), T.int64(1), T.int64(32), T.int64(8)), "float16", scope="warp")
        for threadIdx_z in T.thread_binding(T.int64(2), thread="threadIdx.z"):
            for threadIdx_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for v0 in range(T.int64(4)):
                        for v1 in T.vectorized(T.int64(8)):
                            with T.block("A_reindex_shared.dyn"):
                                T.reads(A[(v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4)) % T.int64(32), threadIdx_x % T.int64(4) * T.int64(8) + v1])
                                T.writes(A_shared_dyn[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), threadIdx_x % T.int64(4) * T.int64(8) + v1])
                                A_shared_dyn[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), T.bitwise_xor(threadIdx_x % T.int64(4), threadIdx_x // T.int64(8)) * T.int64(8) + v1] = A[(v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4)) % T.int64(32), threadIdx_x % T.int64(4) * T.int64(8) + v1]
                    for v0, v1 in T.grid(T.int64(2), T.int64(4)):
                        with T.block("A_reindex_shared.dyn_warp_o"):
                            T.reads(A_shared_dyn[threadIdx_z * T.int64(64) + v1 * T.int64(16):threadIdx_z * T.int64(64) + v1 * T.int64(16) + T.int64(16), v0 * T.int64(16):v0 * T.int64(16) + T.int64(16)])
                            T.writes(A_warp[v1, T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                            with T.block("A_reindex_shared.dyn_warp_o"):
                                T.reads(A_shared_dyn[threadIdx_z * T.int64(64) + v1 * T.int64(16):threadIdx_z * T.int64(64) + v1 * T.int64(16) + T.int64(16), v0 * T.int64(16):v0 * T.int64(16) + T.int64(16)])
                                T.writes(A_warp[v1, T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", A_warp.data, v1 * T.int64(256) + threadIdx_x * T.int64(8), T.tvm_access_ptr(T.type_annotation("float16"), A_shared_dyn.data, threadIdx_z * T.int64(2048) + v1 * T.int64(512) + threadIdx_x % T.int64(16) * T.int64(32) + T.bitwise_xor(v0 * T.int64(2) + threadIdx_x // T.int64(16), threadIdx_x % T.int64(8) // T.int64(2)) * T.int64(8), T.int64(512), 1), T.int64(0))

    # fmt: on
    _check_primfunc_transform(before, expected)


def test_buffer_b():
    # fmt: off
    @T.prim_func
    def before(B: T.Buffer((T.int64(128), T.int64(32)), "float16")):
        B_shared_dyn = T.alloc_buffer((T.int64(128), T.int64(32)), "float16", scope="shared.dyn")
        for threadIdx_z in T.thread_binding(T.int64(2), thread="threadIdx.z"):
            for threadIdx_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for v0 in range(T.int64(4)):
                        for v1 in T.vectorized(T.int64(8)):
                            with T.block("B_reindex_shared.dyn"):
                                T.block_attr({"permuted_layout": 1})
                                B_shared_dyn[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), threadIdx_x % T.int64(4) * T.int64(8) + v1] = B[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), threadIdx_x % T.int64(4) * T.int64(8) + v1]
                    for v0 in range(T.int64(2)):
                        with T.block(""):
                            B_warp = T.alloc_buffer((T.int64(4), T.int64(1), T.int64(32), T.int64(8)), "float16", scope="warp")
                            for v1 in range(T.int64(4)):
                                with T.block("B_reindex_shared.dyn_warp_o"):
                                    T.block_attr({"permuted_layout": 1})
                                    with T.block("B_reindex_shared.dyn_warp_o"):
                                        T.reads(B_shared_dyn[threadIdx_y * T.int64(64) + v1 * T.int64(16):threadIdx_y * T.int64(64) + v1 * T.int64(16) + T.int64(16), v0 * T.int64(16):v0 * T.int64(16) + T.int64(16)])
                                        T.writes(B_warp[v1, T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", B_warp.data, v1 * T.int64(256) + threadIdx_x * T.int64(8), T.tvm_access_ptr(T.type_annotation("float16"), B_shared_dyn.data, threadIdx_y * T.int64(2048) + v1 * T.int64(512) + v0 * T.int64(16), T.int64(512), 1), threadIdx_x // T.int64(16) * T.int64(256) + threadIdx_x % T.int64(8) * T.int64(32) + threadIdx_x % T.int64(16) // T.int64(8) * T.int64(8))

    @T.prim_func
    def expected(B: T.Buffer((T.int64(128), T.int64(32)), "float16")):
        B_shared_dyn = T.alloc_buffer((T.int64(128), T.int64(32)), "float16", scope="shared.dyn")
        for threadIdx_z in T.thread_binding(T.int64(2), thread="threadIdx.z"):
            for threadIdx_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for v0 in range(T.int64(4)):
                        for v1 in T.vectorized(T.int64(8)):
                            with T.block("B_reindex_shared.dyn"):
                                T.reads(B[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), threadIdx_x % T.int64(4) * T.int64(8) + v1])
                                T.writes(B_shared_dyn[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), threadIdx_x % T.int64(4) * T.int64(8) + v1])
                                B_shared_dyn[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), T.bitwise_xor(threadIdx_x % T.int64(4), threadIdx_x // T.int64(8)) * T.int64(8) + v1] = B[v0 * T.int64(32) + threadIdx_z * T.int64(16) + threadIdx_y * T.int64(8) + threadIdx_x // T.int64(4), threadIdx_x % T.int64(4) * T.int64(8) + v1]
                    for v0 in range(T.int64(2)):
                        with T.block(""):
                            B_warp = T.alloc_buffer((T.int64(4), T.int64(1), T.int64(32), T.int64(8)), "float16", scope="warp")
                            for v1 in range(T.int64(4)):
                                with T.block("B_reindex_shared.dyn_warp_o"):
                                    T.reads(B_shared_dyn[threadIdx_y * T.int64(64) + v1 * T.int64(16):threadIdx_y * T.int64(64) + v1 * T.int64(16) + T.int64(16), v0 * T.int64(16):v0 * T.int64(16) + T.int64(16)])
                                    T.writes(B_warp[v1, T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                    with T.block("B_reindex_shared.dyn_warp_o"):
                                        T.reads(B_shared_dyn[threadIdx_y * T.int64(64) + v1 * T.int64(16):threadIdx_y * T.int64(64) + v1 * T.int64(16) + T.int64(16), v0 * T.int64(16):v0 * T.int64(16) + T.int64(16)])
                                        T.writes(B_warp[v1, T.int64(0), T.int64(0):T.int64(32), T.int64(0):T.int64(8)])
                                        T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", B_warp.data, v1 * T.int64(256) + threadIdx_x * T.int64(8), T.tvm_access_ptr(T.type_annotation("float16"), B_shared_dyn.data, threadIdx_y * T.int64(2048) + v1 * T.int64(512) + threadIdx_x // T.int64(16) * T.int64(256) + threadIdx_x % T.int64(8) * T.int64(32) + T.bitwise_xor(v0 * T.int64(2) + threadIdx_x % T.int64(16) // T.int64(8), threadIdx_x % T.int64(8) // T.int64(2)) * T.int64(8), T.int64(512), 1), T.int64(0))

    # fmt: on
    _check_primfunc_transform(before, expected)


def test_buffer_c_fp32():
    # fmt: off
    @T.prim_func
    def before(p_O: T.handle):
        O = T.match_buffer(p_O, (T.int64(128), T.int64(128)), "float16")
        O_shared_dyn = T.alloc_buffer((T.int64(128), T.int64(128)), scope="shared.dyn")
        O_warp = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(32), T.int64(8)), scope="warp")
        for threadIdx_z in T.thread_binding(T.int64(2), thread="threadIdx.z"):
            for threadIdx_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for v0, v1 in T.grid(T.int64(4), T.int64(4)):
                        with T.block("O.dyn_warp_o"):
                            T.block_attr({"permuted_layout": 1})
                            with T.block("O.dyn_warp_o"):
                                for local_id in range(T.int64(8)):
                                    O_shared_dyn[threadIdx_z * T.int64(64) + v0 * T.int64(16) + local_id % T.int64(4) // T.int64(2) * T.int64(8) + threadIdx_x // T.int64(4), threadIdx_y * T.int64(64) + v1 * T.int64(16) + local_id // T.int64(4) * T.int64(8) + threadIdx_x % T.int64(4) * T.int64(2) + local_id % T.int64(2)] = O_warp[v0, v1, threadIdx_x, local_id]
                    for v0 in range(T.int64(16)):
                        for v1 in T.vectorized(T.int64(8)):
                            with T.block("O.dyn"):
                                T.block_attr({"permuted_layout": 1})
                                O[v0 * T.int64(8) + threadIdx_z * T.int64(4) + threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16), threadIdx_x % T.int64(16) * T.int64(8) + v1] = T.Cast("float16", O_shared_dyn[v0 * T.int64(8) + threadIdx_z * T.int64(4) + threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16), threadIdx_x % T.int64(16) * T.int64(8) + v1])


    @T.prim_func
    def expected(O: T.Buffer((T.int64(128), T.int64(128)), "float16")):
        # with T.block("root"):
        O_shared_dyn = T.alloc_buffer((T.int64(128), T.int64(128)), scope="shared.dyn")
        O_warp = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(32), T.int64(8)), scope="warp")
        for threadIdx_z in T.thread_binding(T.int64(2), thread="threadIdx.z"):
            for threadIdx_y in T.thread_binding(T.int64(2), thread="threadIdx.y"):
                for threadIdx_x in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    for v0, v1 in T.grid(T.int64(4), T.int64(4)):
                        with T.block("O.dyn_warp_o"):
                            T.reads(O_warp[v0, v1, threadIdx_x, T.int64(0):T.int64(8)])
                            T.writes(O_shared_dyn[threadIdx_z * T.int64(64) + v0 * T.int64(16) + threadIdx_x // T.int64(4):threadIdx_z * T.int64(64) + v0 * T.int64(16) + threadIdx_x // T.int64(4) + T.int64(9), threadIdx_y * T.int64(64) + v1 * T.int64(16) + threadIdx_x % T.int64(4) * T.int64(2):threadIdx_y * T.int64(64) + v1 * T.int64(16) + threadIdx_x % T.int64(4) * T.int64(2) + T.int64(10)])
                            with T.block("O.dyn_warp_o"):
                                T.reads(O_warp[v0, v1, threadIdx_x, T.int64(0):T.int64(8)])
                                T.writes(O_shared_dyn[threadIdx_z * T.int64(64) + v0 * T.int64(16) + threadIdx_x // T.int64(4):threadIdx_z * T.int64(64) + v0 * T.int64(16) + threadIdx_x // T.int64(4) + T.int64(9), threadIdx_y * T.int64(64) + v1 * T.int64(16) + threadIdx_x % T.int64(4) * T.int64(2):threadIdx_y * T.int64(64) + v1 * T.int64(16) + threadIdx_x % T.int64(4) * T.int64(2) + T.int64(10)])
                                for local_id in range(T.int64(8)):
                                    O_shared_dyn[threadIdx_z * T.int64(64) + v0 * T.int64(16) + local_id % T.int64(4) // T.int64(2) * T.int64(8) + threadIdx_x // T.int64(4), T.bitwise_xor(threadIdx_y * T.int64(8) + v1 * T.int64(2) + local_id // T.int64(4), threadIdx_x // T.int64(4)) * T.int64(8) + threadIdx_x % T.int64(4) * T.int64(2) + local_id % T.int64(2)] = O_warp[v0, v1, threadIdx_x, local_id]
                    for v0 in range(T.int64(16)):
                        for v1 in T.vectorized(T.int64(8)):
                            with T.block("O.dyn"):
                                T.reads(O_shared_dyn[v0 * T.int64(8) + threadIdx_z * T.int64(4) + threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16), threadIdx_x % T.int64(16) * T.int64(8) + v1])
                                T.writes(O[v0 * T.int64(8) + threadIdx_z * T.int64(4) + threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16), threadIdx_x % T.int64(16) * T.int64(8) + v1])
                                O[v0 * T.int64(8) + threadIdx_z * T.int64(4) + threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16), threadIdx_x % T.int64(16) * T.int64(8) + v1] = T.Cast("float16", O_shared_dyn[v0 * T.int64(8) + threadIdx_z * T.int64(4) + threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16), T.bitwise_xor(threadIdx_x % T.int64(16), threadIdx_z * T.int64(4) + threadIdx_y * T.int64(2) + threadIdx_x // T.int64(16)) * T.int64(8) + v1])

    # fmt: on
    _check_primfunc_transform(before, expected)


if __name__ == "__main__":
    tvm.testing.main()
