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
from tvm.script import tir as T


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


@tvm.script.ir_module
class MatmulBefore:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
        # body
        # with T.block("root")
        for blockIdx_y in T.thread_binding(32, thread="blockIdx.y"):
            for blockIdx_x in T.thread_binding(32, thread="blockIdx.x"):
                for threadIdx_y in T.thread_binding(2, thread="threadIdx.y"):
                    for threadIdx_x in T.thread_binding(2, thread="threadIdx.x"):
                        for k_0 in T.serial(32):
                            with T.block():
                                T.reads(A[blockIdx_y * 32 : blockIdx_y * 32 + 32, k_0 * 32 : k_0 * 32 + 32], B[k_0 * 32 : k_0 * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                T.writes(C[blockIdx_y * 32 : blockIdx_y * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                A_shared = T.alloc_buffer([1024, 1024], dtype="float32", scope="shared")
                                B_shared = T.alloc_buffer([1024, 1024], dtype="float32", scope="shared")
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block("A_shared"):
                                            T.reads(A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.writes(A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.block_attr({"tir.manifest_shared_memory_local_stage":1})
                                            A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block("B_shared"):
                                            T.reads(B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.writes(B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.block_attr({"tir.manifest_shared_memory_local_stage":1})
                                            B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for k_1, i_2, j_2, k_2 in T.grid(2, 16, 16, 16):
                                    with T.block("C"):
                                        T.reads(A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2], B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        T.writes(C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        if k_0 * 32 + k_1 * 16 + k_2 == 0:
                                            C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = T.float32(0)
                                        C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] + A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2] * B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2]


@tvm.script.ir_module
class MatmulAfter:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
        # body
        # with T.block("root")
        for blockIdx_y in T.thread_binding(32, thread="blockIdx.y"):
            for blockIdx_x in T.thread_binding(32, thread="blockIdx.x"):
                for threadIdx_y in T.thread_binding(2, thread="threadIdx.y"):
                    for threadIdx_x in T.thread_binding(2, thread="threadIdx.x"):
                        for k_0 in T.serial(32):
                            with T.block():
                                T.reads(A[blockIdx_y * 32 : blockIdx_y * 32 + 32, k_0 * 32 : k_0 * 32 + 32], B[k_0 * 32 : k_0 * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                T.writes(C[blockIdx_y * 32 : blockIdx_y * 32 + 32, blockIdx_x * 32 : blockIdx_x * 32 + 32])
                                A_shared = T.alloc_buffer([1024, 1024], dtype="float32", scope="shared")
                                B_shared = T.alloc_buffer([1024, 1024], dtype="float32", scope="shared")
                                A_shared_local = T.alloc_buffer([64, 4], dtype="float32", scope="local")
                                B_shared_local = T.alloc_buffer([64, 4], dtype="float32", scope="local")
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block():
                                            T.reads(A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.writes(A_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3])
                                            A_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3] = A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block("A_shared"):
                                            T.reads(A[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.writes(A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            A_shared[blockIdx_y * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = A_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block():
                                            T.reads(B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.writes(B_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3])
                                            B_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3] = B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32]
                                for ax0_ax1_fused_0 in T.serial(64):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block("B_shared"):
                                            T.reads(B[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            T.writes(B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32])
                                            B_shared[k_0 * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) // 32, blockIdx_x * 32 + (ax0_ax1_fused_0 * 16 + threadIdx_y * 8 + threadIdx_x * 4 + ax0_ax1_fused_3) % 32] = B_shared_local[ax0_ax1_fused_0, ax0_ax1_fused_3]
                                for k_1, i_2, j_2, k_2 in T.grid(2, 16, 16, 16):
                                    with T.block("C"):
                                        T.reads(A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2], B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        T.writes(C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2])
                                        if k_0 * 32 + k_1 * 16 + k_2 == 0:
                                            C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = T.float32(0)
                                        C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] = C[blockIdx_y * 32 + threadIdx_y * 16 + i_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2] + A_shared[blockIdx_y * 32 + threadIdx_y * 16 + i_2, k_0 * 32 + k_1 * 16 + k_2] * B_shared[k_0 * 32 + k_1 * 16 + k_2, blockIdx_x * 32 + threadIdx_x * 16 + j_2]


# fmt: on
# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def _check(before, expected):
    after = tvm.tir.transform.ManifestSharedMemoryLocalStage()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_transform_matmul():
    _check(MatmulBefore, MatmulAfter)


if __name__ == "__main__":
    tvm.testing.main()
