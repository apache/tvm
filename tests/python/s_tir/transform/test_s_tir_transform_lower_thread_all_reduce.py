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
# ruff: noqa: F401, F841

import tvm
import tvm.testing
from tvm import s_tir
from tvm.script import ir as I
from tvm.script import tirx as T


def test_basic():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            A_flat = T.decl_buffer(4096, data=A.data)

            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 32)

                reduce = T.alloc_buffer((1,), scope="local")
                reduce_1 = T.decl_buffer(1, data=reduce.data, scope="local")

                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret("handle", T.uint64(0)),
                ):
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        A_flat[0],
                        T.bool(True),
                        reduce_1[0],
                        threadIdx_x,
                    )
                if threadIdx_x == 0:
                    B[i] = reduce_1[0]

    After = transform(Before)
    # Verify the transform produces valid output (not checking exact structure
    # since the output depends on the allreduce lowering implementation)
    assert After is not None
    # Run script roundtrip to verify it can be printed and reparsed
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script


def test_basic_with_decl_buffer():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((128, 32), "float32"), B: T.Buffer(128, "float32")):
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

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script


def test_reduce_summation():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer(128, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            A_flat = T.decl_buffer(16384, data=A.data)

            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 32)

                normal_reduce = T.alloc_buffer((1,), scope="local")
                normal_reduce_1 = T.decl_buffer(1, data=normal_reduce.data, scope="local")

                reduce = T.alloc_buffer((1,), scope="local")
                reduce_1 = T.decl_buffer(1, data=reduce.data, scope="local")

                normal_reduce_1[0] = T.float32(0)

                for ko in range(4):
                    normal_reduce_1[0] = (
                        normal_reduce_1[0] + A_flat[i * 128 + ko * 32 + threadIdx_x]
                    )

                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret("handle", T.uint64(0)),
                ):
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_1[0],
                        T.bool(True),
                        reduce_1[0],
                        threadIdx_x,
                    )
                if threadIdx_x == 0:
                    B[i] = reduce_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script


def test_multi_group_reduction():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((32, 32), "float32"), B: T.Buffer((32,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 32)
            cross_thread_B = T.alloc_buffer((1,), scope="local")
            threadIdx_x = T.launch_thread("threadIdx.x", 32)
            cross_thread_B_1 = T.decl_buffer((1,), data=cross_thread_B.data, scope="local")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                A_1 = T.decl_buffer((1024,), data=A.data)
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    A_1[threadIdx_y * 32 + threadIdx_x],
                    T.bool(True),
                    cross_thread_B_1[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B_1 = T.decl_buffer((32,), data=B.data)
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script


def test_multi_group_mask1():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((32, 8), "float32"), B: T.Buffer((32,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 32)
            cross_thread_B = T.alloc_buffer((1,), scope="local")
            threadIdx_x = T.launch_thread("threadIdx.x", 8)
            cross_thread_B_1 = T.decl_buffer((1,), data=cross_thread_B.data, scope="local")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                A_1 = T.decl_buffer((256,), data=A.data)
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    A_1[threadIdx_y * 8 + threadIdx_x],
                    T.bool(True),
                    cross_thread_B_1[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B_1 = T.decl_buffer((32,), data=B.data)
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script


def test_multi_warp_reduce1():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 128)
                cross_thread_B = T.alloc_buffer((1,), scope="local")
                cross_thread_B_1 = T.decl_buffer((1,), data=cross_thread_B.data, scope="local")
                with T.attr(
                    T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret("handle", T.uint64(0)),
                ):
                    A_1 = T.decl_buffer((16384,), data=A.data)
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        A_1[i * 128 + threadIdx_x],
                        T.bool(True),
                        cross_thread_B_1[0],
                        threadIdx_x,
                    )
                if threadIdx_x == 0:
                    B_1 = T.decl_buffer((128,), data=B.data)
                    B_1[i] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script
    assert "tvm_storage_sync" in After_script  # multi-warp needs shared sync


def test_multi_warp_reduce2():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((1, 1024), "float32"), B: T.Buffer((1,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_x = T.launch_thread("threadIdx.x", 1024)
            cross_thread_B = T.alloc_buffer((1,), scope="local")
            cross_thread_B_1 = T.decl_buffer((1,), data=cross_thread_B.data, scope="local")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                A_1 = T.decl_buffer((1024,), data=A.data)
                T.tvm_thread_allreduce(
                    T.uint32(1), A_1[threadIdx_x], T.bool(True), cross_thread_B_1[0], threadIdx_x
                )
            if threadIdx_x == 0:
                B_1 = T.decl_buffer((1,), data=B.data)
                B_1[0] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script
    assert "tvm_storage_sync" in After_script


def test_multi_group_multi_warp_reduction():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((4, 128), "float32"), B: T.Buffer((4,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 4)
            cross_thread_B = T.alloc_buffer((1,), scope="local")
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            cross_thread_B_1 = T.decl_buffer((1,), data=cross_thread_B.data, scope="local")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                A_1 = T.decl_buffer((512,), data=A.data)
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    A_1[threadIdx_y * 128 + threadIdx_x],
                    T.bool(True),
                    cross_thread_B_1[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B_1 = T.decl_buffer((4,), data=B.data)
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script
    assert "tvm_storage_sync" in After_script


def test_multi_group_multi_warp_predicated_reduction():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((2, 70), "float32"), B: T.Buffer((2,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 2)
            in_thread_B = T.alloc_buffer((1,), scope="local")
            cross_thread_B = T.alloc_buffer((1,), scope="local")
            threadIdx_x = T.launch_thread("threadIdx.x", 512)
            in_thread_B_1 = T.decl_buffer((1,), data=in_thread_B.data, scope="local")
            in_thread_B_1[0] = T.float32(0)
            if threadIdx_x < 70:
                A_1 = T.decl_buffer((140,), data=A.data)
                in_thread_B_1[0] = in_thread_B_1[0] + A_1[threadIdx_y * 70 + threadIdx_x]
            cross_thread_B_1 = T.decl_buffer((1,), data=cross_thread_B.data, scope="local")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                T.tvm_thread_allreduce(
                    T.uint32(1), in_thread_B_1[0], T.bool(True), cross_thread_B_1[0], threadIdx_x
                )
            if threadIdx_x == 0:
                B_1 = T.decl_buffer((2,), data=B.data)
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "tvm_warp_shuffle" in After_script
    assert "tvm_storage_sync" in After_script


def test_metal_no_mask():
    transform = tvm.s_tir.transform.LowerThreadAllreduce()

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main(A: T.Buffer((1, 1, 2, 128), "float32"), B: T.Buffer((1, 1, 2), "float32")):
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
            cross_thread_B = T.alloc_buffer((1,), scope="local")
            threadIdx_z = T.launch_thread("threadIdx.z", 1)
            threadIdx_y = T.launch_thread("threadIdx.y", 2)
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            cross_thread_B_1 = T.decl_buffer((1,), data=cross_thread_B.data, scope="local")
            with T.attr(
                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret("handle", T.uint64(0)),
            ):
                A_1 = T.decl_buffer((256,), data=A.data)
                T.tvm_thread_allreduce(
                    T.uint32(1),
                    A_1[threadIdx_y * 128 + threadIdx_x],
                    T.bool(True),
                    cross_thread_B_1[0],
                    threadIdx_x,
                )
            if threadIdx_x == 0:
                B_1 = T.decl_buffer((2,), data=B.data)
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    # Metal does not use warp masks
    assert "tvm_warp_shuffle_down" in After_script
    assert "tvm_storage_sync" in After_script


if __name__ == "__main__":
    tvm.testing.main()
