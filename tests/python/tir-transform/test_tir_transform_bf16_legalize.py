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
import tvm.script
from tvm.target import Target
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.transform.transform import BindTarget


def u16tof32(v):
    uint32_v = v.astype("uint32")
    uint32_v = uint32_v << tvm.tir.const(16, "uint32")
    return T.reinterpret("float32", uint32_v)


def bf16tof32(v):
    return u16tof32(T.reinterpret("uint16", v))


def f32tou16(v):
    uint32_v = T.reinterpret("uint32", v)
    rounding_bias = (uint32_v >> tvm.tir.const(16, "uint32")) & tvm.tir.const(1, "uint32")
    rounding_bias += tvm.tir.const(0x7FFF, "uint32")
    uint32_v = uint32_v + rounding_bias
    return (uint32_v >> tvm.tir.const(16, "uint32")).astype("uint16")


def f32tobf16(v):
    return T.reinterpret("bfloat16", f32tou16(v))


def test_bf16_storage_compute_scope_will_legalize():
    def get_before():
        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
                Bptr: T.handle("bfloat16", storage_scope="local"),
                Dptr: T.handle("bfloat16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "bfloat16", data=Aptr)
                B = T.decl_buffer((100,), "bfloat16", data=Bptr)
                D = T.decl_buffer((100,), "bfloat16", data=Dptr)
                C = T.decl_buffer((100,), "bfloat16")
                for i in T.grid(100):
                    C[i] = A[i] + B[i]
                    D[i] = T.exp(C[i])

        return Before

    def after_compute_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
                Bptr: T.handle("bfloat16", storage_scope="local"),
                Dptr: T.handle("bfloat16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "bfloat16", data=Aptr)
                B = T.decl_buffer((100,), "bfloat16", data=Bptr)
                D = T.decl_buffer((100,), "bfloat16", data=Dptr)
                C = T.decl_buffer((100,), "float32")
                for i in T.grid(100):
                    C[i] = bf16tof32(A[i]) + bf16tof32(B[i])
                    D[i] = f32tobf16(T.exp(C[i]))

        return After

    def after_storage_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func
            def main(
                Aptr: T.handle("uint16", storage_scope="shared"),
                Bptr: T.handle("uint16", storage_scope="local"),
                Dptr: T.handle("uint16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "uint16", data=Aptr)
                B = T.decl_buffer((100,), "uint16", data=Bptr)
                D = T.decl_buffer((100,), "uint16", data=Dptr)
                C = T.decl_buffer((100,), "float32")
                for i in T.grid(100):
                    C[i] = u16tof32(A[i]) + u16tof32(B[i])
                    D[i] = f32tou16(T.exp(C[i]))

        return After

    target = Target("nvidia/geforce-rtx-2080-ti")
    before = BindTarget(target)(get_before())
    after_compute = tvm.tir.transform.BF16ComputeLegalize()(before)
    after_storage = tvm.tir.transform.BF16StorageLegalize()(after_compute)
    tvm.ir.assert_structural_equal(after_compute, BindTarget(target)(after_compute_legalize()))
    tvm.ir.assert_structural_equal(after_storage, BindTarget(target)(after_storage_legalize()))


def test_bf16_storage_compute_scope_wont_legalize():
    def get_before():
        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
                Bptr: T.handle("bfloat16", storage_scope="local"),
                Dptr: T.handle("bfloat16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "bfloat16", data=Aptr)
                B = T.decl_buffer((100,), "bfloat16", data=Bptr)
                D = T.decl_buffer((100,), "bfloat16", data=Dptr)
                C = T.decl_buffer((100,), "bfloat16")
                for i in T.grid(100):
                    C[i] = A[i] + B[i]
                    D[i] = T.exp(C[i])

        return Before

    def after_compute_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
                Bptr: T.handle("bfloat16", storage_scope="local"),
                Dptr: T.handle("bfloat16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "bfloat16", data=Aptr)
                B = T.decl_buffer((100,), "bfloat16", data=Bptr)
                D = T.decl_buffer((100,), "bfloat16", data=Dptr)
                C = T.decl_buffer((100,), "bfloat16")
                for i in T.grid(100):
                    C[i] = A[i] + B[i]
                    D[i] = T.exp(C[i])

        return After

    def after_storage_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
                Bptr: T.handle("bfloat16", storage_scope="local"),
                Dptr: T.handle("bfloat16"),
            ):
                T.func_attr({"global_symbol": "main"})
                A = T.decl_buffer((100,), "bfloat16", data=Aptr)
                B = T.decl_buffer((100,), "bfloat16", data=Bptr)
                D = T.decl_buffer((100,), "bfloat16", data=Dptr)
                C = T.decl_buffer((100,), "bfloat16")
                for i in T.grid(100):
                    C[i] = A[i] + B[i]
                    D[i] = T.exp(C[i])

        return After

    target = Target("nvidia/geforce-rtx-3090-ti")
    before = BindTarget(target)(get_before())
    after_compute = tvm.tir.transform.BF16ComputeLegalize()(before)
    after_storage = tvm.tir.transform.BF16StorageLegalize()(after_compute)
    tvm.ir.assert_structural_equal(after_compute, BindTarget(target)(after_compute_legalize()))
    tvm.ir.assert_structural_equal(after_storage, BindTarget(target)(after_storage_legalize()))


def test_bf16_reduce_will_legalize():
    def get_before():
        @tvm.script.ir_module
        class Before:
            @T.prim_func(private=True)
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
            ):
                A_flat = T.decl_buffer(4096, "bfloat16", data=Aptr)

                for i in range(128):
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)

                    reduce = T.decl_buffer(1, dtype="bfloat16", scope="local")

                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.bfloat16(0)]),
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

        return Before

    def after_compute_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func(private=True)
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
            ):
                A_flat_1 = T.decl_buffer(4096, "bfloat16", data=Aptr)

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
                            T.reinterpret(
                                "float32",
                                T.shift_left(
                                    T.Cast("uint32", T.reinterpret("uint16", A_flat_1[0])),
                                    T.uint32(16),
                                ),
                            ),
                            T.bool(True),
                            reduce[0],
                            threadIdx_x,
                        )

        return After

    def after_storage_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func(private=True)
            def main(
                Aptr: T.handle("uint16", storage_scope="shared"),
            ):
                A_flat_1 = T.decl_buffer(4096, "uint16", data=Aptr)

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
                            T.reinterpret(
                                "float32",
                                T.shift_left(
                                    T.Cast("uint32", T.reinterpret("uint16", A_flat_1[0])),
                                    T.uint32(16),
                                ),
                            ),
                            T.bool(True),
                            reduce[0],
                            threadIdx_x,
                        )

        return After

    target = Target("nvidia/geforce-rtx-2080-ti")
    before = BindTarget(target)(get_before())
    after_compute = tvm.tir.transform.BF16ComputeLegalize()(before)
    after_storage = tvm.tir.transform.BF16StorageLegalize()(after_compute)
    tvm.ir.assert_structural_equal(after_compute, BindTarget(target)(after_compute_legalize()))
    tvm.ir.assert_structural_equal(after_storage, BindTarget(target)(after_storage_legalize()))


def test_bf16_reduce_wont_legalize():
    def get_before():
        @tvm.script.ir_module
        class Before:
            @T.prim_func(private=True)
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
            ):
                A_flat = T.decl_buffer(4096, "bfloat16", data=Aptr)

                for i in range(128):
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)

                    reduce = T.decl_buffer(1, dtype="bfloat16", scope="local")

                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.bfloat16(0)]),
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

        return Before

    def after_compute_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func(private=True)
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
            ):
                A_flat = T.decl_buffer(4096, "bfloat16", data=Aptr)

                for i in range(128):
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)

                    reduce = T.decl_buffer(1, dtype="bfloat16", scope="local")

                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.bfloat16(0)]),
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

        return After

    def after_storage_legalize():
        @tvm.script.ir_module
        class After:
            @T.prim_func(private=True)
            def main(
                Aptr: T.handle("bfloat16", storage_scope="shared"),
            ):
                A_flat = T.decl_buffer(4096, "bfloat16", data=Aptr)

                for i in range(128):
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)

                    reduce = T.decl_buffer(1, dtype="bfloat16", scope="local")

                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.bfloat16(0)]),
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

        return After

    target = Target("nvidia/geforce-rtx-3090-ti")
    before = BindTarget(target)(get_before())
    after_compute = tvm.tir.transform.BF16ComputeLegalize()(before)
    after_storage = tvm.tir.transform.BF16StorageLegalize()(after_compute)
    tvm.ir.assert_structural_equal(after_compute, BindTarget(target)(after_compute_legalize()))
    tvm.ir.assert_structural_equal(after_storage, BindTarget(target)(after_storage_legalize()))


if __name__ == "__main__":
    test_bf16_storage_compute_scope_will_legalize()
    test_bf16_storage_compute_scope_wont_legalize()
    test_bf16_reduce_will_legalize()
    test_bf16_reduce_wont_legalize()
