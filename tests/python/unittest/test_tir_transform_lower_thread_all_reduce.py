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
    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
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
                mask = T.decl_buffer(1, "uint32", data=mask_data, scope="local")

                t0_data = T.allocate([1], "float32", "local")
                t0 = T.decl_buffer(1, data=t0_data, scope="local")

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
    @T.prim_func(private=True)
    def before(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.decl_buffer(4096, data=A.data)

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

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.decl_buffer(4096, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            reduce = T.decl_buffer(1, dtype="float32", scope="local")

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                mask_data = T.allocate([1], "uint32", "local")
                mask = T.decl_buffer(1, "uint32", data=mask_data, scope="local")

                t0_data = T.allocate([1], "float32", "local")
                t0 = T.decl_buffer(1, data=t0_data, scope="local")

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
    @T.prim_func(private=True)
    def before(A: T.Buffer((128, 128), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.decl_buffer(16384, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            normal_reduce_data = T.allocate([1], "float32", "local")
            normal_reduce = T.decl_buffer(1, data=normal_reduce_data, scope="local")

            reduce_data = T.allocate([1], "float32", "local")
            reduce = T.decl_buffer(1, data=reduce_data, scope="local")

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

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128, 128), "float32"), B: T.Buffer(128, "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        A_flat = T.decl_buffer(16384, data=A.data)

        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 32)

            normal_reduce_data = T.allocate([1], "float32", "local")
            normal_reduce = T.decl_buffer(1, data=normal_reduce_data, scope="local")

            reduce_data = T.allocate([1], "float32", "local")
            reduce = T.decl_buffer(1, data=reduce_data, scope="local")

            normal_reduce[0] = T.float32(0)
            for ko in range(4):
                normal_reduce[0] = normal_reduce[0] + A_flat[i * 128 + ko * 32 + threadIdx_x]
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                mask_data = T.allocate([1], "uint32", "local")
                mask = T.decl_buffer(1, "uint32", data=mask_data, scope="local")

                t0_data = T.allocate([1], "float32", "local")
                t0 = T.decl_buffer(1, data=t0_data, scope="local")

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


class TestMultiGroupReduction(BaseCompare):
    @T.prim_func(private=True)
    def before(A: T.Buffer((32, 32), "float32"), B: T.Buffer((32,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 32)
        cross_thread_B = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        cross_thread_B_1 = T.Buffer((1,), data=cross_thread_B, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            A_1 = T.Buffer((1024,), data=A.data)
            T.tvm_thread_allreduce(
                T.uint32(1),
                A_1[threadIdx_y * 32 + threadIdx_x],
                T.bool(True),
                cross_thread_B_1[0],
                threadIdx_x,
            )
        if threadIdx_x == 0:
            B_1 = T.Buffer((32,), data=B.data)
            B_1[threadIdx_y] = cross_thread_B_1[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((32, 32), "float32"), B: T.Buffer((32,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 32)
        red_buf0 = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        red_buf0_1 = T.Buffer((1,), data=red_buf0, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            mask = T.decl_buffer([1], "uint32", scope="local")
            t0 = T.decl_buffer([1], "float32", scope="local")
            A_1 = T.Buffer((1024,), data=A.data)
            red_buf0_1[0] = A_1[threadIdx_y * 32 + threadIdx_x]

            mask[0] = T.tvm_warp_activemask()

            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            red_buf0_1[0] = T.tvm_warp_shuffle(mask[0], red_buf0_1[0], 32 * threadIdx_y, 32, 32)
        if threadIdx_x == 0:
            B_1 = T.Buffer((32,), data=B.data)
            B_1[threadIdx_y] = red_buf0_1[0]


class TestMultiGroupMask1(BaseCompare):
    @T.prim_func(private=True)
    def before(A: T.Buffer((32, 8), "float32"), B: T.Buffer((32,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 32)
        cross_thread_B = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 8)
        cross_thread_B_1 = T.Buffer((1,), data=cross_thread_B, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            A_1 = T.Buffer((256,), data=A.data)
            T.tvm_thread_allreduce(
                T.uint32(1),
                A_1[threadIdx_y * 8 + threadIdx_x],
                T.bool(True),
                cross_thread_B_1[0],
                threadIdx_x,
            )
        if threadIdx_x == 0:
            B_1 = T.Buffer((32,), data=B.data)
            B_1[threadIdx_y] = cross_thread_B_1[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((32, 8), "float32"), B: T.Buffer((32,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 32)
        red_buf0 = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 8)
        red_buf0_1 = T.Buffer((1,), data=red_buf0, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            mask = T.decl_buffer([1], "uint32", scope="local")
            t0 = T.decl_buffer([1], "float32", scope="local")
            A_1 = T.Buffer((256,), data=A.data)
            red_buf0_1[0] = A_1[threadIdx_y * 8 + threadIdx_x]
            mask[0] = T.bitwise_and(
                T.tvm_warp_activemask(),
                T.shift_left(T.uint32(255), T.uint32(8) * T.Cast("uint32", threadIdx_y)),
            )
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            red_buf0_1[0] = T.tvm_warp_shuffle(mask[0], red_buf0_1[0], 8 * threadIdx_y, 32, 32)
        if threadIdx_x == 0:
            B_1 = T.Buffer((32,), data=B.data)
            B_1[threadIdx_y] = red_buf0_1[0]


class TestMultiWarpReduce1(BaseCompare):
    @T.prim_func(private=True)
    def before(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            cross_thread_B = T.allocate([1], "float32", "local")
            cross_thread_B_1 = T.Buffer((1,), data=cross_thread_B, scope="local")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                A_1 = T.Buffer((16384,), data=A.data)
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    A_1[i * 128 + threadIdx_x],
                    T.bool(True),
                    cross_thread_B_1[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B_1 = T.Buffer((128,), data=B.data)
                B_1[i] = cross_thread_B_1[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        for i in range(128):
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            red_result = T.allocate([1], "float32", "shared")
            T.attr(red_result, "volatile_scope", 1)
            red_result_1 = T.Buffer((1,), data=red_result, scope="shared")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                red_buf0 = T.decl_buffer([1], "float32", scope="local")
                mask = T.decl_buffer([1], "uint32", scope="local")
                t0 = T.decl_buffer([1], "float32", scope="local")
                red_buf0_1 = T.decl_buffer([1], "float32", scope="local")
                mask_1 = T.decl_buffer([1], "uint32", scope="local")
                t0_1 = T.decl_buffer([1], "float32", scope="local")
                red_buf_staging = T.decl_buffer([4], "float32", scope="shared")
                A_1 = T.Buffer((16384,), data=A.data)
                red_buf0_1[0] = A_1[i * 128 + threadIdx_x]
                mask_1[0] = T.tvm_warp_activemask()
                t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 16, 32, 32)
                red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
                t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 8, 32, 32)
                red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
                t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 4, 32, 32)
                red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
                t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 2, 32, 32)
                red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
                t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 1, 32, 32)
                red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
                if threadIdx_x % 32 == 0:
                    red_buf_staging[threadIdx_x // 32] = red_buf0_1[0]
                T.tvm_storage_sync("shared")
                if threadIdx_x < 4:
                    red_buf0[0] = red_buf_staging[threadIdx_x]
                mask[0] = T.bitwise_and(T.tvm_warp_activemask(), T.uint32(15))
                t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 2, 32, 32)
                red_buf0[0] = red_buf0[0] + t0[0]
                t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 1, 32, 32)
                red_buf0[0] = red_buf0[0] + t0[0]
                if threadIdx_x == 0:
                    red_result_1[0] = red_buf0[0]
                T.tvm_storage_sync("shared")
            if threadIdx_x == 0:
                B_1 = T.Buffer((128,), data=B.data)
                B_1[i] = red_result_1[0]


class TestMultiWarpReduce2(BaseCompare):
    @T.prim_func(private=True)
    def before(A: T.Buffer((1, 1024), "float32"), B: T.Buffer((1,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_x = T.launch_thread("threadIdx.x", 1024)
        cross_thread_B = T.allocate([1], "float32", "local")
        cross_thread_B_1 = T.Buffer((1,), data=cross_thread_B, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            A_1 = T.Buffer((1024,), data=A.data)
            T.tvm_thread_allreduce(
                T.uint32(1), A_1[threadIdx_x], T.bool(True), cross_thread_B_1[0], threadIdx_x
            )
        if threadIdx_x == 0:
            B_1 = T.Buffer((1,), data=B.data)
            B_1[0] = cross_thread_B_1[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((1, 1024), "float32"), B: T.Buffer((1,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_x = T.launch_thread("threadIdx.x", 1024)
        red_result = T.allocate([1], "float32", "shared")
        T.attr(red_result, "volatile_scope", 1)
        red_result_1 = T.Buffer((1,), data=red_result, scope="shared")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            red_buf0 = T.decl_buffer([1], "float32", scope="local")
            mask = T.decl_buffer([1], "uint32", scope="local")
            t0 = T.decl_buffer([1], "float32", scope="local")
            red_buf0_1 = T.decl_buffer([1], "float32", scope="local")
            mask_1 = T.decl_buffer([1], "uint32", scope="local")
            t0_1 = T.decl_buffer([1], "float32", scope="local")
            red_buf_staging = T.decl_buffer([32], "float32", scope="shared")
            A_1 = T.Buffer((1024,), data=A.data)
            red_buf0_1[0] = A_1[threadIdx_x]
            mask_1[0] = T.tvm_warp_activemask()
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            if threadIdx_x % 32 == 0:
                red_buf_staging[threadIdx_x // 32] = red_buf0_1[0]
            T.tvm_storage_sync("shared")
            if threadIdx_x < 32:
                red_buf0[0] = red_buf_staging[threadIdx_x]
            mask[0] = T.tvm_warp_activemask()
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 16, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 8, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 4, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 2, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 1, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            if threadIdx_x == 0:
                red_result_1[0] = red_buf0[0]
            T.tvm_storage_sync("shared")
        if threadIdx_x == 0:
            B_1 = T.Buffer((1,), data=B.data)
            B_1[0] = red_result_1[0]


class TestMultiGroupMultiWarpReduction(BaseCompare):
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 128), "float32"), B: T.Buffer((4,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 4)
        cross_thread_B = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        cross_thread_B_1 = T.Buffer((1,), data=cross_thread_B, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            A_1 = T.Buffer((512,), data=A.data)
            T.tvm_thread_allreduce(
                T.uint32(1),
                A_1[threadIdx_y * 128 + threadIdx_x],
                T.bool(True),
                cross_thread_B_1[0],
                threadIdx_x,
            )
        if threadIdx_x == 0:
            B_1 = T.Buffer((4,), data=B.data)
            B_1[threadIdx_y] = cross_thread_B_1[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 128), "float32"), B: T.Buffer((4,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 4)
        red_result = T.allocate([4], "float32", "shared")
        T.attr(red_result, "volatile_scope", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        red_result_1 = T.Buffer((4,), data=red_result, scope="shared")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            red_buf0 = T.decl_buffer([1], "float32", scope="local")
            mask = T.decl_buffer([1], "uint32", scope="local")
            t0 = T.decl_buffer([1], "float32", scope="local")
            red_buf0_1 = T.decl_buffer([1], "float32", scope="local")
            mask_1 = T.decl_buffer([1], "uint32", scope="local")
            t0_1 = T.decl_buffer([1], "float32", scope="local")
            red_buf_staging = T.decl_buffer([16], "float32", scope="shared")
            A_1 = T.Buffer((512,), data=A.data)
            red_buf0_1[0] = A_1[threadIdx_y * 128 + threadIdx_x]
            mask_1[0] = T.tvm_warp_activemask()
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            if threadIdx_x % 32 == 0:
                red_buf_staging[threadIdx_y * 4 + threadIdx_x // 32] = red_buf0_1[0]
            T.tvm_storage_sync("shared")
            if threadIdx_x < 4:
                red_buf0[0] = red_buf_staging[threadIdx_y * 4 + threadIdx_x]
            mask[0] = T.bitwise_and(
                T.tvm_warp_activemask(), T.Cast("uint32", T.shift_left(15, threadIdx_y * 4))
            )
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 2, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 1, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            if threadIdx_x == 0:
                red_result_1[threadIdx_y] = red_buf0[0]
            T.tvm_storage_sync("shared")
        if threadIdx_x == 0:
            B_1 = T.Buffer((4,), data=B.data)
            B_1[threadIdx_y] = red_result_1[threadIdx_y]


class TestMultiGroupMultiWarpPredicatedReduction(BaseCompare):
    @T.prim_func(private=True)
    def before(A: T.Buffer((2, 70), "float32"), B: T.Buffer((2,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 2)
        in_thread_B = T.allocate([1], "float32", "local")
        cross_thread_B = T.allocate([1], "float32", "local")
        threadIdx_x = T.launch_thread("threadIdx.x", 512)
        in_thread_B_1 = T.Buffer((1,), data=in_thread_B, scope="local")
        in_thread_B_1[0] = T.float32(0)
        if threadIdx_x < 70:
            A_1 = T.Buffer((140,), data=A.data)
            in_thread_B_1[0] = in_thread_B_1[0] + A_1[threadIdx_y * 70 + threadIdx_x]
        cross_thread_B_1 = T.Buffer((1,), data=cross_thread_B, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            T.tvm_thread_allreduce(
                T.uint32(1), in_thread_B_1[0], T.bool(True), cross_thread_B_1[0], threadIdx_x
            )
        if threadIdx_x == 0:
            B_1 = T.Buffer((2,), data=B.data)
            B_1[threadIdx_y] = cross_thread_B_1[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((2, 70), "float32"), B: T.Buffer((2,), "float32")):
        T.func_attr({"target": T.target("cuda", host="llvm")})
        threadIdx_y = T.launch_thread("threadIdx.y", 2)
        in_thread_B = T.allocate([1], "float32", "local")
        red_result = T.allocate([2], "float32", "shared")
        T.attr(red_result, "volatile_scope", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 512)
        in_thread_B_1 = T.Buffer((1,), data=in_thread_B, scope="local")
        in_thread_B_1[0] = T.float32(0)
        if threadIdx_x < 70:
            A_1 = T.Buffer((140,), data=A.data)
            in_thread_B_1[0] = in_thread_B_1[0] + A_1[threadIdx_y * 70 + threadIdx_x]
        red_result_1 = T.Buffer((2,), data=red_result, scope="shared")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            red_buf0 = T.decl_buffer([1], "float32", scope="local")
            mask = T.decl_buffer([1], "uint32", scope="local")
            t0 = T.decl_buffer([1], "float32", scope="local")
            red_buf0_1 = T.decl_buffer([1], "float32", scope="local")
            mask_1 = T.decl_buffer([1], "uint32", scope="local")
            t0_1 = T.decl_buffer([1], "float32", scope="local")
            red_buf_staging = T.decl_buffer([32], "float32", scope="shared")
            red_buf0_1[0] = in_thread_B_1[0]
            mask_1[0] = T.tvm_warp_activemask()
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            if threadIdx_x % 32 == 0:
                red_buf_staging[threadIdx_y * 16 + threadIdx_x // 32] = red_buf0_1[0]
            T.tvm_storage_sync("shared")
            if threadIdx_x < 16:
                red_buf0[0] = red_buf_staging[threadIdx_y * 16 + threadIdx_x]
            mask[0] = T.bitwise_and(
                T.tvm_warp_activemask(), T.Cast("uint32", T.shift_left(65535, threadIdx_y * 16))
            )
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 8, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 4, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 2, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0[0], 1, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            if threadIdx_x == 0:
                red_result_1[threadIdx_y] = red_buf0[0]
            T.tvm_storage_sync("shared")
        if threadIdx_x == 0:
            B_1 = T.Buffer((2,), data=B.data)
            B_1[threadIdx_y] = red_result_1[threadIdx_y]


class TestMetalNoMask(BaseCompare):
    @T.prim_func(private=True)
    def before(A: T.Buffer((1, 1, 2, 128), "float32"), B: T.Buffer((1, 1, 2), "float32")):
        T.func_attr(
            {
                "target": T.target(
                    {
                        "kind": "metal",
                        "max_threads_per_block": 1024,
                        "thread_warp_size": 32,
                        "host": "llvm",
                    }
                ),
            }
        )
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        cross_thread_B = T.allocate([1], "float32", "local")
        threadIdx_z = T.launch_thread("threadIdx.z", 1)
        threadIdx_y = T.launch_thread("threadIdx.y", 2)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        cross_thread_B_1 = T.Buffer((1,), data=cross_thread_B, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            A_1 = T.Buffer((256,), data=A.data)
            T.tvm_thread_allreduce(
                T.uint32(1),
                A_1[threadIdx_y * 128 + threadIdx_x],
                T.bool(True),
                cross_thread_B_1[0],
                threadIdx_x,
            )
        if threadIdx_x == 0:
            B_1 = T.Buffer((2,), data=B.data)
            B_1[threadIdx_y] = cross_thread_B_1[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer((1, 1, 2, 128), "float32"), B: T.Buffer((1, 1, 2), "float32")):
        T.func_attr(
            {
                "target": T.target(
                    {
                        "kind": "metal",
                        "max_threads_per_block": 1024,
                        "thread_warp_size": 32,
                        "host": "llvm",
                    }
                ),
            }
        )
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        red_result = T.allocate([2], "float32", "shared")
        T.attr(red_result, "volatile_scope", 1)
        threadIdx_z = T.launch_thread("threadIdx.z", 1)
        threadIdx_y = T.launch_thread("threadIdx.y", 2)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        red_result_1 = T.Buffer((2,), data=red_result, scope="shared")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret("handle", T.uint64(0)),
        ):
            red_buf0 = T.decl_buffer([1], "float32", scope="local")
            t0 = T.decl_buffer([1], "float32", scope="local")
            red_buf0_1 = T.decl_buffer([1], "float32", scope="local")
            t0_1 = T.decl_buffer([1], "float32", scope="local")
            red_buf_staging = T.decl_buffer([8], "float32", scope="shared")
            A_1 = T.Buffer((256,), data=A.data)
            red_buf0_1[0] = A_1[threadIdx_y * 128 + threadIdx_x]
            t0_1[0] = T.tvm_warp_shuffle_down(0, red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(0, red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(0, red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(0, red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(0, red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            if threadIdx_x % 32 == 0:
                red_buf_staging[threadIdx_y * 4 + threadIdx_x // 32] = red_buf0_1[0]
            T.tvm_storage_sync("shared")
            if threadIdx_x < 4:
                red_buf0[0] = red_buf_staging[threadIdx_y * 4 + threadIdx_x]
            t0[0] = T.tvm_warp_shuffle_down(0, red_buf0[0], 2, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(0, red_buf0[0], 1, 32, 32)
            red_buf0[0] = red_buf0[0] + t0[0]
            if threadIdx_x == 0:
                red_result_1[threadIdx_y] = red_buf0[0]
            T.tvm_storage_sync("shared")
        if threadIdx_x == 0:
            B_1 = T.Buffer((2,), data=B.data)
            B_1[threadIdx_y] = red_result_1[threadIdx_y]


if __name__ == "__main__":
    tvm.testing.main()
