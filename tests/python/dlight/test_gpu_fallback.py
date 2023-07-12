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
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


def test_fallback():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(
            A: T.Buffer((1, 32, 1, 128), "float16"),
            C: T.Buffer((1, 1, 4096), "float16"),
        ):
            B = T.alloc_buffer((1, 1, 32, 128), "float16")
            for i, j, k, l in T.grid(1, 1, 32, 128):
                with T.block("T_transpose"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk, vl] = A[vi, vk, vj, vl]
            for i, j, k in T.grid(1, 1, 4096):
                with T.block("T_reshape"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = B[0, 0, vk % 4096 // 128, vk % 128]

    @I.ir_module
    class After:
        @T.prim_func
        def main(
            A: T.Buffer((1, 32, 1, 128), "float16"),
            C: T.Buffer((1, 1, 4096), "float16"),
        ):
            T.func_attr({"tir.is_scheduled": 1})
            for ax0_fused_0 in T.thread_binding(4, thread="blockIdx.x"):
                for ax0_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                    with T.block("T_reshape"):
                        v0 = T.axis.spatial(4096, ax0_fused_0 * 1024 + ax0_fused_1)
                        T.reads(A[0, v0 // 128, 0, v0 % 128])
                        T.writes(C[0, 0, v0])
                        C[0, 0, v0] = A[0, v0 // 128, 0, v0 % 128]

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.Fallback(),
        )(Before)
    assert_structural_equal(mod, After)


def test_fallback_reduction():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((1, 6144), "float32"), B: T.Buffer((1,), "float32")):
            for ax0, ax1 in T.grid(1, 6144):
                with T.block("block"):
                    v0 = T.axis.spatial(1, ax0)
                    v1 = T.axis.reduce(6144, ax1)
                    T.reads(A[v0, v1])
                    T.writes(B[v0])
                    with T.init():
                        B[v0] = T.float32(0)
                    B[v0] = B[v0] + T.Cast("float32", A[v0, v1])

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((1, 6144), "float32"), B: T.Buffer((1,), "float32")):
            T.func_attr({"tir.is_scheduled": 1})
            for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    with T.block("block_init"):
                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(1))
                        T.reads()
                        T.writes(B[0])
                        B[0] = T.float32(0)
                    for ax1 in range(6144):
                        with T.block("block_update"):
                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v1 = T.axis.reduce(6144, ax1)
                            T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(1))
                            T.reads(B[0], A[0, v1])
                            T.writes(B[0])
                            B[0] = B[0] + T.Cast("float32", A[0, v1])

    with Target("apple/m1-gpu"):
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.Fallback(),
        )(Module)
    assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
