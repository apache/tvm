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
from tvm import te
from tvm.script import tir as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(
        mod["main"], transformed.with_attr("global_symbol", "main"), True
    )


@T.prim_func
def compacted_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer([1, 16], "float32", scope="global")
            for j in range(0, 16):
                with T.block():
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block():
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@T.prim_func
def transformed_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in T.serial(0, 16):
        B_new = T.decl_buffer(shape=[1, 16], dtype="float32")
        for j in T.serial(0, 16):
            B_new[0, j] = A[i, j] + 1.0
        for j in T.serial(0, 16):
            C[i, j] = B_new[0, j] * 2.0


@T.prim_func
def compacted_gpu_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.thread_binding(0, 4, thread="blockIdx.x"):
        for i1 in T.thread_binding(0, 2, thread="threadIdx.x"):
            for i2 in T.thread_binding(0, 2, thread="vthread"):
                with T.block():
                    T.reads(A[i0 * 4 + i1 * 2 + i2, 0:16])
                    T.writes(C[i0 * 4 + i1 * 2 + i2, 0:16])
                    B = T.alloc_buffer([1, 16], "float32", scope="local")
                    for j in range(0, 16):
                        with T.block():
                            T.reads(A[i0 * 4 + i1 * 2 + i2, j])
                            T.writes(B[0, j])
                            B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block():
                            T.reads(B[0, j])
                            T.writes(C[i0 * 4 + i1 * 2 + i2, j])
                            C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0


@T.prim_func
def transformed_gpu_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")

    i0 = T.env_thread("blockIdx.x")
    i1 = T.env_thread("threadIdx.x")
    i2 = T.env_thread("vthread")

    T.launch_thread(i0, 4)
    T.launch_thread(i1, 2)
    T.launch_thread(i2, 2)
    B = T.decl_buffer(shape=[1, 16], dtype="float32", scope="local")
    for j in range(0, 16):
        B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
    for j in range(0, 16):
        C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0


@T.prim_func
def compacted_symbolic_func(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
    A = T.match_buffer(a, (n, m), "float32")
    C = T.match_buffer(c, (n, m), "float32")

    for i in range(0, n):
        with T.block():
            T.reads(A[i, m])
            T.writes(C[i, m])
            B = T.alloc_buffer((m,), "float32", scope="global")
            for j in range(0, m):
                with T.block():
                    T.reads(A[i, j])
                    T.writes(B[j])
                    B[j] = A[i, j] + 1.0
            for j in range(0, m):
                with T.block():
                    T.reads(B[j])
                    T.writes(C[i, j])
                    C[i, j] = B[j] * 2.0


@T.prim_func
def transformed_symbolic_func(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
    A = T.match_buffer(a, (n, m), "float32")
    C = T.match_buffer(c, (n, m), "float32")

    for i in range(0, n):
        B = T.decl_buffer(shape=[m], dtype="float32")
        for j in range(0, m):
            B[j] = A[i, j] + 1.0
        for j in range(0, m):
            C[i, j] = B[j] * 2.0


@T.prim_func
def compacted_predicate_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for i, j in T.grid(5, 7):
        with T.block():
            T.reads(A[i * 7 + j])
            T.writes(C[i * 7 + j])
            T.where(i * 7 + j < 32)
            C[i * 7 + j] = A[i * 7 + j] + 1.0


@T.prim_func
def transformed_predicate_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for i, j in T.grid(5, 7):
        if i * 7 + j < 32:
            C[i * 7 + j] = A[i * 7 + j] + 1.0


@T.prim_func
def compacted_unit_loop_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for x, y, z in T.grid(4, 1, 8):
        with T.block():
            T.reads(A[x * 8 + y * 8 + z])
            T.writes(C[x * 8 + y * 8 + z])
            C[x * 8 + y * 8 + z] = A[x * 8 + y * 8 + z] + 1.0


@T.prim_func
def transformed_unit_loop_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for x, z in T.grid(4, 8):
        C[x * 8 + z] = A[x * 8 + z] + 1.0


@T.prim_func
def compacted_multi_alloc_func(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    D = T.match_buffer(d, (32), "float32")

    for i in range(0, 32):
        with T.block():
            T.reads(A[i])
            T.writes(D[i])
            B = T.alloc_buffer((32,), scope="global")
            C = T.alloc_buffer((32,), scope="global")
            B[i] = A[i] + 1.0
            C[i] = A[i] + B[i]
            D[i] = C[i] * 2.0


@T.prim_func
def transformed_multi_alloc_func(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    D = T.match_buffer(d, (32), "float32")

    for i in range(0, 32):
        B = T.decl_buffer(shape=(32,), dtype="float32")
        C = T.decl_buffer(shape=(32,), dtype="float32")
        B[i] = A[i] + 1.0
        C[i] = A[i] + B[i]
        D[i] = C[i] * 2.0


@T.prim_func
def compacted_strided_buffer_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in range(0, 4):
        with T.block():
            T.reads(A[i0 * 4 : i0 * 4 + 4, 0:16])
            T.writes(C[i0 * 4 : i0 * 4 + 4, 0:16])
            B = T.alloc_buffer([4, 16], "float32", strides=[17, 1], scope="global")
            for i1 in range(0, 4):
                for j in range(0, 16):
                    with T.block():
                        T.reads(A[i0 * 4 + i1, j])
                        T.writes(B[i1, j])
                        B[i1, j] = A[i0 * 4 + i1, j] + 1.0
            for i1 in range(0, 4):
                for j in range(0, 16):
                    with T.block():
                        T.reads(B[i1, j])
                        T.writes(C[i0 * 4 + i1, j])
                        C[i0 * 4 + i1, j] = B[i1, j] * 2.0


@T.prim_func
def transformed_strided_buffer_func(
    A: T.Buffer((16, 16), "float32"), C: T.Buffer((16, 16), "float32")
) -> None:
    # body
    for i0 in T.serial(4):
        B_data = T.allocate([4, 17], "float32", "global")
        B = T.decl_buffer(shape=[4, 16], dtype="float32", strides=[17, 1], data=B_data)
        for i1, j in T.grid(4, 16):
            B[i1, j] = A[i0 * 4 + i1, j] + T.float32(1)
        for i1, j in T.grid(4, 16):
            C[i0 * 4 + i1, j] = B[i1, j] * T.float32(2)


@T.prim_func
def compacted_symbolic_strided_buffer_func(a: T.handle) -> None:
    n = T.int32()
    A = T.match_buffer(a, (1, n, 10240))
    padded_size = T.meta_var(T.min((n + 63) // 64 * 64, 96))
    # with T.block("root"):
    for i, j, k in T.grid(((n + 63) // 64 * 4 + 7) // 8, 2, 160):
        with T.block(""):
            A_pad_shared_dyn = T.alloc_buffer(
                (1, padded_size, 64), strides=(72 * padded_size, 72, 1), scope="shared.dyn"
            )
            for ax0, ax1 in T.grid(96, 64):
                with T.block("A_pad_shared.dyn"):
                    T.where(i * 128 + j * 32 + ax0 < (n + 63) // 64 * 64)
                    A_pad_shared_dyn[0, ax0, ax1] = T.if_then_else(
                        i * 128 + j * 32 + ax0 < n,
                        A[0, i * 128 + j * 32 + ax0, k * 64 + ax1],
                        T.float32(0),
                    )


@T.prim_func
def transformed_symbolic_strided_buffer_func(a: T.handle):
    n = T.int32()
    A = T.match_buffer(a, (1, n, 10240))
    for i, j, k in T.grid(((n + 63) // 64 * 4 + 7) // 8, 2, 160):
        A_pad_shared_dyn = T.allocate(
            [1, T.min((n + 63) // 64 * 64, 96), 72], "float32", "shared.dyn"
        )
        A_pad_shared_dyn_1 = T.decl_buffer(
            (1, T.min((n + 63) // 64 * 64, 96), 64),
            data=A_pad_shared_dyn,
            strides=(72 * T.min((n + 63) // 64 * 64, 96), 72, 1),
            scope="shared.dyn",
        )
        for ax0, ax1 in T.grid(96, 64):
            if i * 128 + j * 32 + ax0 < (n + 63) // 64 * 64:
                A_pad_shared_dyn_1[0, ax0, ax1] = T.if_then_else(
                    i * 128 + j * 32 + ax0 < n,
                    A[0, i * 128 + j * 32 + ax0, k * 64 + ax1],
                    T.float32(0),
                )


@T.prim_func
def annotated_loops(a: T.handle) -> None:
    A = T.match_buffer(a, (16,), "float32")
    for i in range(0, 16, annotations={"pragma_1": "str_value", "pragma_2": 1, "pragma_3": 0.0}):
        A[i] = 0.0


@T.prim_func
def boolean_handling_before(a: T.Buffer(10, "bool"), b: T.Buffer(10, "bool")) -> None:
    for i0 in T.serial(10):
        with T.block("b"):
            T.reads(a[i0])
            T.writes(b[i0])
            b[i0] = a[i0]


@T.prim_func
def boolean_handling_after(a: T.Buffer(10, "bool"), b: T.Buffer(10, "bool")) -> None:
    # body
    for i0 in T.serial(10):
        b[i0] = a[i0]


def test_elementwise():
    _check(compacted_elementwise_func, transformed_elementwise_func)


def test_gpu_workload():
    _check(compacted_gpu_func, transformed_gpu_func)


def test_symbolic_shape():
    _check(compacted_symbolic_func, transformed_symbolic_func)


def test_predicate():
    _check(compacted_predicate_func, transformed_predicate_func)


def test_unit_loops():
    _check(compacted_unit_loop_func, transformed_unit_loop_func)


def test_multi_alloc():
    _check(compacted_multi_alloc_func, transformed_multi_alloc_func)


def test_strided_buffer():
    _check(compacted_strided_buffer_func, transformed_strided_buffer_func)


def test_symbolic_strided_buffer():
    _check(compacted_symbolic_strided_buffer_func, transformed_symbolic_strided_buffer_func)


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.LowerOpaqueBlock()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # LowerOpaqueBlock should do nothing on TE


def test_annotated_loops():
    mod = tvm.IRModule.from_expr(annotated_loops.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    attr1 = mod["main"].body
    attr2 = attr1.body
    attr3 = attr2.body
    assert attr1.attr_key == "pragma_1" and attr1.value == "str_value"
    assert attr2.attr_key == "pragma_2"
    tvm.ir.assert_structural_equal(attr2.value, tvm.tir.IntImm("int32", 1))
    assert attr3.attr_key == "pragma_3"
    tvm.ir.assert_structural_equal(attr3.value, tvm.tir.FloatImm("float32", 0.0))


def test_annotated_block():
    @T.prim_func
    def annotated_block() -> None:
        with T.block():
            T.block_attr({"pragma_1": "str_value", "pragma_2": 1, "pragma_3": 0.0})
            T.evaluate(0)

    mod = tvm.IRModule.from_expr(annotated_block.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    attr1 = mod["main"].body
    attr2 = attr1.body
    attr3 = attr2.body
    assert attr1.attr_key == "pragma_1" and attr1.value == "str_value"
    assert attr2.attr_key == "pragma_2"
    tvm.ir.assert_structural_equal(attr2.value, tvm.tir.IntImm("int32", 1))
    assert attr3.attr_key == "pragma_3"
    tvm.ir.assert_structural_equal(attr3.value, tvm.tir.FloatImm("float32", 0.0))


def test_preserved_annotations():
    @T.prim_func
    def before(A: T.Buffer(8, "float32"), B: T.Buffer(8, "float32")):
        for i in T.serial(8, annotations={"k_0": 1, "k_1": [2, 3], "k_2": 3.14}):
            with T.block("block"):
                T.block_attr({"k_3": "oops"})
                B[i] = A[i] + 1.0

    @T.prim_func
    def after(A: T.Buffer(8, "float32"), B: T.Buffer(8, "float32")):
        for i in T.serial(8, annotations={"k_0": 1, "k_1": [2, 3], "k_2": 3.14}):
            B[i] = A[i] + 1.0

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    tvm.ir.assert_structural_equal(mod["main"], after.with_attr("global_symbol", "main"))


def test_boolean_handling():
    _check(boolean_handling_before, boolean_handling_after)


if __name__ == "__main__":
    tvm.testing.main()
