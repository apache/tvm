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


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.transform.Sequential(
        [
            tvm.tir.transform.UnifyKernelLaunch(),
            tvm.tir.transform.Simplify(),
            tvm.tir.transform.RemoveNoOp(),
        ]
    )


class TestElementwise(BaseCompare):
    @T.prim_func
    def before(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j0_0 in T.thread_binding(0, 4, "threadIdx.x"):
                for j0_1 in T.serial(0, 32):
                    with T.block(""):
                        B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j1_0 in T.thread_binding(0, 4, "threadIdx.x"):
                for j1_1 in T.serial(0, 32):
                    with T.block(""):
                        C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0

    @T.prim_func
    def expected(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])

        for blockIdx_x in T.thread_binding(0, 128, "blockIdx.x"):
            for threadIdx_x in T.thread_binding(0, 4, "threadIdx.x"):
                for j0_1 in T.serial(0, 32):
                    with T.block(""):
                        B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                            A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                        )
                for j1_1 in T.serial(0, 32):
                    with T.block(""):
                        C[blockIdx_x, threadIdx_x * 32 + j1_1] = (
                            B[blockIdx_x, threadIdx_x * 32 + j1_1] + 1.0
                        )


class TestNoFuseWithSingleOp(BaseCompare):
    @T.prim_func
    def before(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j0_0 in T.thread_binding(0, 4, "threadIdx.x"):
                for j0_1 in T.serial(0, 32):
                    with T.block(""):
                        B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0

    expected = before


class TestNoFuseWithDifferentBinding1(BaseCompare):
    @T.prim_func
    def before(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j0_0 in T.thread_binding(0, 4, "threadIdx.x"):
                for j0_1 in T.serial(0, 32):
                    with T.block(""):
                        B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j1_0 in T.thread_binding(0, 5, "threadIdx.x"):
                for j1_1 in T.serial(0, 32):
                    with T.block(""):
                        C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0

    expected = before


class TestNoFuseWithDifferentBinding2(BaseCompare):
    @T.prim_func
    def before(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j0_0 in T.thread_binding(0, 4, "threadIdx.x"):
                for j0_1 in T.serial(0, 32):
                    with T.block(""):
                        B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j1_0 in T.thread_binding(0, 4, "threadIdx.x"):
                for j1_1 in T.thread_binding(0, 32, "vthread.x"):
                    with T.block(""):
                        C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0

    expected = before


class TestComplex(BaseCompare):
    @T.prim_func
    def before(A: T.Buffer([128, 128], "float32"), B: T.Buffer([128, 128], "float32")):
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                with T.block():
                    B[i, j] = A[i, j] * 2.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                B[i, j] = B[i, j] + 1.0
        bx = T.env_thread("blockIdx.x")
        tx = T.env_thread("threadIdx.x")
        with T.launch_thread(bx, 128):
            with T.launch_thread(tx, 128):
                B[bx, tx] = B[bx, tx] + 1.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j0 in T.thread_binding(0, 16, "threadIdx.x"):
                for j1 in T.thread_binding(0, 8, "threadIdx.y"):
                    B[i, j0 * 8 + j1] = B[i, j0 * 8 + j1] + 1.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                B[i, j] = B[i, j] + 1.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                B[i, j] = B[i, j] + 2.0

    @T.prim_func
    def expected(A: T.Buffer([128, 128], "float32"), B: T.Buffer([128, 128], "float32")):
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                with T.block():
                    B[i, j] = A[i, j] * 2.0
                B[i, j] = B[i, j] + 1.0
                B[i, j] = B[i, j] + 1.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j0 in T.thread_binding(0, 16, "threadIdx.x"):
                for j1 in T.thread_binding(0, 8, "threadIdx.y"):
                    B[i, j0 * 8 + j1] = B[i, j0 * 8 + j1] + 1.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                B[i, j] = B[i, j] + 1.0
                B[i, j] = B[i, j] + 2.0


class TestNoFuseWithShared(BaseCompare):
    @T.prim_func
    def before(A: T.Buffer([128, 128], "float32"), B: T.Buffer([128, 128], "float32")):
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                with T.block():
                    B[i, j] = A[i, j] * 2.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                with T.block():
                    tmp = T.alloc_buffer([128, 128], dtype="float32", scope="shared.dyn")
                    B[i, j] = B[i, j] + 1.0

    expected = before


class TestFuseWithSharedInFirstKernel(BaseCompare):
    @T.prim_func
    def before(A: T.Buffer([128, 128], "float32"), B: T.Buffer([128, 128], "float32")):
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                with T.block():
                    tmp = T.alloc_buffer([128, 128], dtype="float32", scope="shared.dyn")
                    B[i, j] = A[i, j] * 2.0
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                with T.block():
                    B[i, j] = B[i, j] + 1.0

    @T.prim_func
    def expected(A: T.Buffer([128, 128], "float32"), B: T.Buffer([128, 128], "float32")):
        for i in T.thread_binding(0, 128, "blockIdx.x"):
            for j in T.thread_binding(0, 128, "threadIdx.x"):
                with T.block():
                    tmp = T.alloc_buffer([128, 128], dtype="float32", scope="shared.dyn")
                    B[i, j] = A[i, j] * 2.0
                with T.block():
                    B[i, j] = B[i, j] + 1.0


if __name__ == "__main__":
    tvm.testing.main()
