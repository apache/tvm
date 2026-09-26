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
# ruff: noqa: E501, F401
import tvm
from tvm import s_tir, tirx
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_lift_tx_beyond_local():
    # fmt: off
    n = T.dynamic("n", "int32")

    @Ts.prim_func
    def before(A: T.Buffer((32, 1, 128)), B: T.Buffer((32, n, 128)), C: T.Buffer((32, 1, n))):

        for ax0_ax1_fused in T.thread_binding(n * 32, thread="blockIdx.x"):
            with Ts.sblock(""):
                Ts.reads(A[ax0_ax1_fused // n, 0, 0:256], B[ax0_ax1_fused // n, ax0_ax1_fused % n, 0:256])
                Ts.writes(C[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                D_local = Ts.sblock_alloc_buffer((32, 1, n), scope="local")
                D_rf_local = Ts.sblock_alloc_buffer((256, 32, 1, n), scope="local")
                for ax2_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                    with Ts.sblock("NT_matmul_rf_init"):
                        Ts.reads()
                        Ts.writes(D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                        D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = T.float32(0)
                    for ax2_fused_0 in range(1):
                        with Ts.sblock("NT_matmul_rf_update"):
                            Ts.where(ax2_fused_0 * 256 + ax2_fused_1 < 128)
                            Ts.reads(D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n], A[ax0_ax1_fused // n, 0, ax2_fused_0 * 256 + ax2_fused_1], B[ax0_ax1_fused // n, ax0_ax1_fused % n, ax2_fused_0 * 256 + ax2_fused_1])
                            Ts.writes(D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                            D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = D_rf_local[ax2_fused_1, ax0_ax1_fused // n, 0, ax0_ax1_fused % n] + A[ax0_ax1_fused // n, 0, ax2_fused_0 * 256 + ax2_fused_1] * B[ax0_ax1_fused // n, ax0_ax1_fused % n, ax2_fused_0 * 256 + ax2_fused_1]
                for ax1_ax2_fused in range(1):
                    for ax0_fused in T.thread_binding(256, thread="threadIdx.x"):
                        with Ts.sblock(""):
                            Ts.reads(D_rf_local[ax0_fused, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                            Ts.writes(D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                            cross_thread_D_local = Ts.sblock_alloc_buffer((1,), strides=(1,), scope="local")
                            in_thread_D_local = Ts.sblock_alloc_buffer((1,), strides=(1,), scope="local")
                            with Ts.sblock("NT_matmul_in_thread_init"):
                                Ts.reads()
                                Ts.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = T.float32(0)
                            with Ts.sblock("NT_matmul_in_thread"):
                                Ts.where(0 <= ax0_ax1_fused // n and ax0_ax1_fused // n < 32 and 0 <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
                                Ts.reads(D_rf_local[ax0_fused, ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                                Ts.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = in_thread_D_local[0] + D_rf_local[ax0_fused, ax0_ax1_fused // n, 0, ax0_ax1_fused % n]
                            with Ts.sblock("NT_matmul_cross_thread"):
                                Ts.reads(in_thread_D_local[0])
                                Ts.writes(cross_thread_D_local[0])
                                T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]), "reduce_scope", T.int32(0))
                                T.tvm_thread_allreduce(T.uint32(1), in_thread_D_local[0], T.bool(True), cross_thread_D_local[0], ax0_fused)
                            with Ts.sblock("NT_matmul_write_back"):
                                Ts.where(ax0_fused == 0)
                                Ts.reads(cross_thread_D_local[0])
                                Ts.writes(D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                                D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = cross_thread_D_local[0]
                with Ts.sblock("T_divide"):
                    Ts.where(0 <= ax0_ax1_fused // n and ax0_ax1_fused // n < 32 and 0 <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
                    Ts.reads(D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                    Ts.writes(C[ax0_ax1_fused // n, 0, ax0_ax1_fused % n])
                    C[ax0_ax1_fused // n, 0, ax0_ax1_fused % n] = D_local[ax0_ax1_fused // n, 0, ax0_ax1_fused % n] * T.float32(0.088397790055248615)

    n = T.dynamic("n", "int32")

    @Ts.prim_func
    def expected(A: T.Buffer((32, 1, 128), "float32"), B: T.Buffer((32, n, 128)), C: T.Buffer((32, 1, n))):

        # with Ts.sblock("root"):
        for blockIdx_x in T.thread_binding(n * 32, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(256, thread="threadIdx.x"):
                with Ts.sblock(""):
                    Ts.reads(A[blockIdx_x // n, 0, 0:256], B[blockIdx_x // n, blockIdx_x % n, 0:256])
                    Ts.writes(C[blockIdx_x // n, 0, blockIdx_x % n])
                    D_local = Ts.sblock_alloc_buffer((32, 1, n), scope="local")
                    D_rf_local = Ts.sblock_alloc_buffer((256, 32, 1, n), scope="local")
                    with Ts.sblock("NT_matmul_rf_init"):
                        Ts.reads()
                        Ts.writes(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                        D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n] = T.float32(0)
                    for ax2_fused_0 in range(1):
                        with Ts.sblock("NT_matmul_rf_update"):
                            Ts.where(ax2_fused_0 * 256 + threadIdx_x < 128)
                            Ts.reads(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n], A[blockIdx_x // n, 0, ax2_fused_0 * 256 + threadIdx_x], B[blockIdx_x // n, blockIdx_x % n, ax2_fused_0 * 256 + threadIdx_x])
                            Ts.writes(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                            D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n] = D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n] + A[blockIdx_x // n, 0, ax2_fused_0 * 256 + threadIdx_x] * B[blockIdx_x // n, blockIdx_x % n, ax2_fused_0 * 256 + threadIdx_x]
                    for ax1_ax2_fused in range(1):
                        with Ts.sblock(""):
                            Ts.reads(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                            Ts.writes(D_local[blockIdx_x // n, 0, blockIdx_x % n])
                            cross_thread_D_local = Ts.sblock_alloc_buffer((1,), strides=(1,), scope="local")
                            in_thread_D_local = Ts.sblock_alloc_buffer((1,), strides=(1,), scope="local")
                            with Ts.sblock("NT_matmul_in_thread_init"):
                                Ts.reads()
                                Ts.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = T.float32(0)
                            with Ts.sblock("NT_matmul_in_thread"):
                                Ts.where(0 <= blockIdx_x // n and blockIdx_x // n < 32 and 0 <= blockIdx_x % n and blockIdx_x % n < n)
                                Ts.reads(D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n])
                                Ts.writes(in_thread_D_local[0])
                                in_thread_D_local[0] = in_thread_D_local[0] + D_rf_local[threadIdx_x, blockIdx_x // n, 0, blockIdx_x % n]
                            with Ts.sblock("NT_matmul_cross_thread"):
                                Ts.reads(in_thread_D_local[0])
                                Ts.writes(cross_thread_D_local[0])
                                T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]), "reduce_scope", T.int32(0))
                                T.tvm_thread_allreduce(T.uint32(1), in_thread_D_local[0], T.bool(True), cross_thread_D_local[0], threadIdx_x)
                            with Ts.sblock("NT_matmul_write_back"):
                                Ts.where(threadIdx_x == 0)
                                Ts.reads(cross_thread_D_local[0])
                                Ts.writes(D_local[blockIdx_x // n, 0, blockIdx_x % n])
                                D_local[blockIdx_x // n, 0, blockIdx_x % n] = cross_thread_D_local[0]
                    with Ts.sblock("T_divide"):
                        Ts.where(0 <= blockIdx_x // n and blockIdx_x // n < 32 and 0 <= blockIdx_x % n and blockIdx_x % n < n)
                        Ts.reads(D_local[blockIdx_x // n, 0, blockIdx_x % n])
                        Ts.writes(C[blockIdx_x // n, 0, blockIdx_x % n])
                        C[blockIdx_x // n, 0, blockIdx_x % n] = D_local[blockIdx_x // n, 0, blockIdx_x % n] * T.float32(0.088397790055248615)
    # fmt: on
    mod = tvm.IRModule({"main": before.with_attr("global_symbol", "main")})
    after = s_tir.transform.LiftThreadBinding()(mod)
    tvm.ir.assert_structural_equal(expected.with_attr("global_symbol", "main"), after["main"])


if __name__ == "__main__":
    test_lift_tx_beyond_local()
