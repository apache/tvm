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
from tvm import tir
from tvm.script import tir as T


def test_lift_tx_beyond_local():
    # fmt: off
    @T.prim_func
    def before(a: T.handle, b: T.handle, c: T.handle):
        n = T.int32()
        A = T.match_buffer(a, (32, 1, 128))
        B = T.match_buffer(b, (32, n, 128))
        C = T.match_buffer(c, (32, 1, n))
        for ax0_ax1_fused in T.thread_binding(n * 32, thread="blockIdx.x"):
            with T.block(""):
                T.reads(A[ax0_ax1_fused // n, 0, 0:256], B[ax0_ax1_fused // n, ax0_ax1_fused % n, 0:256])
                T.writes(C[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                D_local = T.alloc_buffer((32, 1, n), scope="local")
                D_rf_local = T.alloc_buffer((256, 32, 1, n), scope="local")
                for ax2_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                    with T.block("NT_matmul_rf_init"):
                        T.reads()
                        T.writes(D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                        D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = T.float32(0)
                    for ax2_fused_0 in range(1):
                        with T.block("NT_matmul_rf_update"):
                            T.where(ax2_fused_0 * 256 + ax2_fused_1 < 128)
                            T.reads(D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n], A[ax0_ax1_fused // n, 0, ax2_fused_0 * 256 + ax2_fused_1], B[ax0_ax1_fused // n, ax0_ax1_fused % n, ax2_fused_0 * 256 + ax2_fused_1])
                            T.writes(D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                            D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n] + A[ax0_ax1_fused // n, 0, ax2_fused_0 * 256 + ax2_fused_1] * B[ax0_ax1_fused // n, ax0_ax1_fused % n, ax2_fused_0 * 256 + ax2_fused_1]
                for ax1_ax2_fused in range(1):
                    for ax0_fused in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block(""):
                            T.reads(D_rf_local[ax0_fused, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                            T.writes(D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                            cross_thread_D_local = T.alloc_buffer((1,), strides=(1,), scope="local")
                            in_thread_D_local = T.alloc_buffer((1,), strides=(1,), scope="local")
                            with T.block("NT_matmul_in_thread_init"):
                                T.reads()
                                T.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = T.float32(0)
                            with T.block("NT_matmul_in_thread"):
                                T.where(0 <= ax0_ax1_fused // n and ax0_ax1_fused // n < 32 and 0 <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
                                T.reads(D_rf_local[ax0_fused, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                                T.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = in_thread_D_local[0] + D_rf_local[ax0_fused, ax0_ax1_fused // n, 0, ax0_ax1_fused % n]
                            with T.block("NT_matmul_cross_thread"):
                                T.reads(in_thread_D_local[0])
                                T.writes(cross_thread_D_local[0])
                                T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0)))
                                T.tvm_thread_allreduce(T.uint32(1), in_thread_D_local[0], T.bool(True), cross_thread_D_local[0], ax0_fused)
                            with T.block("NT_matmul_write_back"):
                                T.where(ax0_fused == 0)
                                T.reads(cross_thread_D_local[0])
                                T.writes(D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                                D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = cross_thread_D_local[0]
                with T.block("T_divide"):
                    T.where(0 <= ax0_ax1_fused // n and ax0_ax1_fused // n < 32 and 0 <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
                    T.reads(D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                    T.writes(C[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                    C[ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n] * T.float32(0.088397790055248615)

    @T.prim_func
    def expected(A: T.Buffer((32, 1, 128), "float32"), b: T.handle, c: T.handle):
        n = T.int32()
        B = T.match_buffer(b, (32, n, 128))
        C = T.match_buffer(c, (32, 1, n))
        # with T.block("root"):
        for blockIdx_x in T.thread_binding(n * 32, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(256, thread="threadIdx.x"):
                with T.block(""):
                    T.reads(A[blockIdx_x // n, 0, 0:256], B[blockIdx_x // n, blockIdx_x % n, 0:256])
                    T.writes(C[blockIdx_x // n, 0, blockIdx_x % n])
                    D_local = T.alloc_buffer((32, 1, n), scope="local")
                    D_rf_local = T.alloc_buffer((256, 32, 1, n), scope="local")
                    with T.block("NT_matmul_rf_init"):
                        T.reads()
                        T.writes(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                        D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n] = T.float32(0)
                    for ax2_fused_0 in range(1):
                        with T.block("NT_matmul_rf_update"):
                            T.where(ax2_fused_0 * 256 + threadIdx_x < 128)
                            T.reads(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n], A[blockIdx_x // n, 0, ax2_fused_0 * 256 + threadIdx_x], B[blockIdx_x // n, blockIdx_x % n, ax2_fused_0 * 256 + threadIdx_x])
                            T.writes(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                            D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n] = D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n] + A[blockIdx_x // n, 0, ax2_fused_0 * 256 + threadIdx_x] * B[blockIdx_x // n, blockIdx_x % n, ax2_fused_0 * 256 + threadIdx_x]
                    for ax1_ax2_fused in range(1):
                        with T.block(""):
                            T.reads(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                            T.writes(D_local[blockIdx_x // n, 0, blockIdx_x % n])
                            cross_thread_D_local = T.alloc_buffer((1,), strides=(1,), scope="local")
                            in_thread_D_local = T.alloc_buffer((1,), strides=(1,), scope="local")
                            with T.block("NT_matmul_in_thread_init"):
                                T.reads()
                                T.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = T.float32(0)
                            with T.block("NT_matmul_in_thread"):
                                T.where(0 <= blockIdx_x // n and blockIdx_x // n < 32 and 0 <= blockIdx_x % n and blockIdx_x % n < n)
                                T.reads(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                                T.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = in_thread_D_local[0] + D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n]
                            with T.block("NT_matmul_cross_thread"):
                                T.reads(in_thread_D_local[0])
                                T.writes(cross_thread_D_local[0])
                                T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0)))
                                T.tvm_thread_allreduce(T.uint32(1), in_thread_D_local[0], T.bool(True), cross_thread_D_local[0], threadIdx_x)
                            with T.block("NT_matmul_write_back"):
                                T.where(threadIdx_x == 0)
                                T.reads(cross_thread_D_local[0])
                                T.writes(D_local[blockIdx_x // n, 0, blockIdx_x % n])
                                D_local[blockIdx_x // n, 0, blockIdx_x % n] = cross_thread_D_local[0]
                    with T.block("T_divide"):
                        T.where(0 <= blockIdx_x // n and blockIdx_x // n < 32 and 0 <= blockIdx_x % n and blockIdx_x % n < n)
                        T.reads(D_local[blockIdx_x // n, 0, blockIdx_x % n])
                        T.writes(C[blockIdx_x // n, 0, blockIdx_x % n])
                        C[blockIdx_x // n, 0, blockIdx_x % n] = D_local[blockIdx_x // n, 0, blockIdx_x % n] * T.float32(0.088397790055248615)
    # fmt: on
    mod = tvm.IRModule({"main": before.with_attr("global_symbol", "main")})
    after = tir.transform.LiftThreadBinding()(mod)
    tvm.ir.assert_structural_equal(expected.with_attr("global_symbol", "main"), after["main"])


if __name__ == "__main__":
    test_lift_tx_beyond_local()
