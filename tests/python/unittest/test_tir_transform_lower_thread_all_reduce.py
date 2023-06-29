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
    transform = tvm.tir.transform.LowerThreadAllreduce()


class BaseFailure(BaseCompare):
    expected = ValueError


class TestBasic(BaseCompare):
    def before(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.Buffer(4096, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            reduce_data = T.allocate([1], "float32", "local")
            reduce = T.Buffer(1, data=reduce_data, scope="local")

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    A_flat[0],
                    T.bool(True),
                    reduce[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B[i] = reduce[0]

    def expected(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.Buffer(4096, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            reduce_data = T.allocate([1], "float32", "local")
            reduce = T.Buffer(1, data=reduce_data, scope="local")

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                mask_data = T.allocate([1], "uint32", "local")
                mask = T.Buffer(1, "uint32", data=mask_data, scope="local")

                t0_data = T.allocate([1], "float32", "local")
                t0 = T.Buffer(1, data=t0_data, scope="local")

                reduce[0] = A_flat[0]
                mask[0] = T.tvm_warp_activemask()

                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 16, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 8, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 4, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 2, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 1, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                reduce[0] = T.tvm_warp_shuffle(mask[0], reduce[0], 0, 32, 32)
            if threadIdx_x == 0:
                B[i] = reduce[0]


class TestBasicWithDeclBuffer(BaseCompare):
    def before(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.Buffer(4096, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            reduce = T.decl_buffer(1, dtype="float32", scope="local")

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    A_flat[0],
                    T.bool(True),
                    reduce[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B[i] = reduce[0]

    def expected(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.Buffer(4096, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            reduce = T.decl_buffer(1, dtype="float32", scope="local")

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                mask_data = T.allocate([1], "uint32", "local")
                mask = T.Buffer(1, "uint32", data=mask_data, scope="local")

                t0_data = T.allocate([1], "float32", "local")
                t0 = T.Buffer(1, data=t0_data, scope="local")

                reduce[0] = A_flat[0]
                mask[0] = T.tvm_warp_activemask()

                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 16, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 8, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 4, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 2, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 1, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                reduce[0] = T.tvm_warp_shuffle(mask[0], reduce[0], 0, 32, 32)
            if threadIdx_x == 0:
                B[i] = reduce[0]


class TestReduceSummation(BaseCompare):
    def before(A: T.Buffer((128, 128), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.Buffer((16384,), data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            normal_reduce_data = T.allocate([1], "float32", "local")
            normal_reduce = T.Buffer(1, data=normal_reduce_data, scope="local")

            reduce_data = T.allocate([1], "float32", "local")
            reduce = T.Buffer(1, data=reduce_data, scope="local")

            normal_reduce[0] = T.float32(0)

            for ko in range(4):
                normal_reduce[0] = normal_reduce[0] + A_flat[i * 128 + ko * 32 + threadIdx_x]

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    normal_reduce[0],
                    T.bool(True),
                    reduce[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B[i] = reduce[0]

    def expected(A: T.Buffer((128, 128), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.Buffer(16384, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            normal_reduce_data = T.allocate([1], "float32", "local")
            normal_reduce = T.Buffer(1, data=normal_reduce_data, scope="local")

            reduce_data = T.allocate([1], "float32", "local")
            reduce = T.Buffer(1, data=reduce_data, scope="local")

            normal_reduce[0] = T.float32(0)
            for ko in range(4):
                normal_reduce[0] = normal_reduce[0] + A_flat[i * 128 + ko * 32 + threadIdx_x]
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                mask_data = T.allocate([1], "uint32", "local")
                mask = T.Buffer(1, "uint32", data=mask_data, scope="local")

                t0_data = T.allocate([1], "float32", "local")
                t0 = T.Buffer(1, data=t0_data, scope="local")

                reduce[0] = normal_reduce[0]
                mask[0] = T.tvm_warp_activemask()

                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 16, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 8, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 4, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 2, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], reduce[0], 1, 32, 32)
                reduce[0] = reduce[0] + t0[0]
                reduce[0] = T.tvm_warp_shuffle(mask[0], reduce[0], 0, 32, 32)
            if threadIdx_x == 0:
                B[i] = reduce[0]


if __name__ == "__main__":
    tvm.testing.main()
