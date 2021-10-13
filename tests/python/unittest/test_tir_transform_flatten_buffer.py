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
from tvm import tir, te
from tvm.script import tir as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


@T.prim_func
def compacted_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block([]):
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer([1, 16], "float32", scope="global")
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@T.prim_func
def flattened_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in T.serial(0, 16):
        B_new = T.allocate([16], "float32", "global")
        for j in T.serial(0, 16):
            B_new[j] = T.load("float32", A.data, ((i * 16) + j)) + 1.0
        for j in T.serial(0, 16):
            C.data[((i * 16) + j)] = T.load("float32", B_new, j) * 2.0


@T.prim_func
def compacted_gpu_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.thread_binding(0, 4, thread="blockIdx.x"):
        for i1 in T.thread_binding(0, 2, thread="threadIdx.x"):
            for i2 in T.thread_binding(0, 2, thread="vthread"):
                with T.block([]):
                    T.reads(A[i0 * 4 + i1 * 2 + i2, 0:16])
                    T.writes(C[i0 * 4 + i1 * 2 + i2, 0:16])
                    B = T.alloc_buffer([1, 16], "float32", scope="local")
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(A[i0 * 4 + i1 * 2 + i2, j])
                            T.writes(B[0, j])
                            B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(B[0, j])
                            T.writes(C[i0 * 4 + i1 * 2 + i2, j])
                            C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0


@T.prim_func
def flattened_gpu_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")

    i0 = T.env_thread("blockIdx.x")
    i1 = T.env_thread("threadIdx.x")
    i2 = T.env_thread("vthread")

    T.launch_thread(i0, 4)
    T.launch_thread(i1, 2)
    T.launch_thread(i2, 2)
    B = T.allocate([16], "float32", "local")
    for j in range(0, 16):
        B[j] = T.load("float32", A.data, i0 * 64 + i1 * 32 + i2 * 16 + j) + 1.0
    for j in range(0, 16):
        C.data[i0 * 64 + i1 * 32 + i2 * 16 + j] = T.load("float32", B, j) * 2.0


@T.prim_func
def compacted_symbolic_func(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
    A = T.match_buffer(a, (n, m), "float32")
    C = T.match_buffer(c, (n, m), "float32")

    for i in range(0, n):
        with T.block([]):
            T.reads(A[i, m])
            T.writes(C[i, m])
            B = T.alloc_buffer((m,), "float32", scope="global")
            for j in range(0, m):
                with T.block([]) as []:
                    T.reads(A[i, j])
                    T.writes(B[j])
                    B[j] = A[i, j] + 1.0
            for j in range(0, m):
                with T.block([]) as []:
                    T.reads(B[j])
                    T.writes(C[i, j])
                    C[i, j] = B[j] * 2.0


@T.prim_func
def flattened_symbolic_func(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
    A = T.match_buffer(a, (n, m), "float32")
    C = T.match_buffer(c, (n, m), "float32")

    for i in range(0, n):
        B = T.allocate([m], "float32", "global")
        for j in range(0, m):
            B[j] = T.load("float32", A.data, i * m + j) + 1.0
        for j in range(0, m):
            C.data[i * m + j] = T.load("float32", B, j) * 2.0


@T.prim_func
def compacted_predicate_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for i, j in T.grid(5, 7):
        with T.block([]) as []:
            T.reads(A[i * 7 + j])
            T.writes(C[i * 7 + j])
            T.where(i * 7 + j < 32)
            C[i * 7 + j] = A[i * 7 + j] + 1.0


@T.prim_func
def flattened_predicate_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for i, j in T.grid(5, 7):
        if i * 7 + j < 32:
            C.data[i * 7 + j] = T.load("float32", A.data, i * 7 + j) + 1.0


@T.prim_func
def compacted_unit_loop_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for x, y, z in T.grid(4, 1, 8):
        with T.block([]) as []:
            T.reads(A[x * 8 + y * 8 + z])
            T.writes(C[x * 8 + y * 8 + z])
            C[x * 8 + y * 8 + z] = A[x * 8 + y * 8 + z] + 1.0


@T.prim_func
def flattened_unit_loop_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    C = T.match_buffer(c, (32), "float32")

    for x, z in T.grid(4, 8):
        C.data[x * 8 + z] = T.load("float32", A.data, x * 8 + z) + 1.0


@T.prim_func
def compacted_multi_alloc_func(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    D = T.match_buffer(d, (32), "float32")

    for i in range(0, 32):
        with T.block([]) as []:
            T.reads(A[i])
            T.writes(D[i])
            B = T.alloc_buffer((32,), scope="global")
            C = T.alloc_buffer((32,), scope="global")
            B[i] = A[i] + 1.0
            C[i] = A[i] + B[i]
            D[i] = C[i] * 2.0


@T.prim_func
def flattened_multi_alloc_func(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (32), "float32")
    D = T.match_buffer(d, (32), "float32")

    for i in range(0, 32):
        B = T.allocate((32,), "float32", "global")
        C = T.allocate((32,), "float32", "global")
        B[i] = T.load("float32", A.data, i) + 1.0
        C[i] = T.load("float32", A.data, i) + T.load("float32", B, i)
        D.data[i] = T.load("float32", C, i) * 2.0


@T.prim_func
def compacted_strided_buffer_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in range(0, 4):
        with T.block([]):
            T.reads(A[i0 * 4 : i0 * 4 + 4, 0:16])
            T.writes(C[i0 * 4 : i0 * 4 + 4, 0:16])
            B = T.alloc_buffer([4, 16], "float32", strides=[17, 1], scope="global")
            for i1 in range(0, 4):
                for j in range(0, 16):
                    with T.block() as []:
                        T.reads(A[i0 * 4 + i1, j])
                        T.writes(B[i1, j])
                        B[i1, j] = A[i0 * 4 + i1, j] + 1.0
            for i1 in range(0, 4):
                for j in range(0, 16):
                    with T.block() as []:
                        T.reads(B[i1, j])
                        T.writes(C[i0 * 4 + i1, j])
                        C[i0 * 4 + i1, j] = B[i1, j] * 2.0


@T.prim_func
def flattened_strided_buffer_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.serial(0, 4):
        B_new = T.allocate([68], "float32", "global")
        for i1 in T.serial(0, 4):
            for j in T.serial(0, 16):
                B_new[i1 * 17 + j] = T.load("float32", A.data, i0 * 64 + i1 * 16 + j) + 1.0
        for i1 in T.serial(0, 4):
            for j in T.serial(0, 16):
                C.data[i0 * 64 + i1 * 16 + j] = T.load("float32", B_new, i1 * 17 + j) * 2.0


def test_elementwise():
    _check(compacted_elementwise_func, flattened_elementwise_func)


def test_gpu_workload():
    _check(compacted_gpu_func, flattened_gpu_func)


def test_symbolic_shape():
    _check(compacted_symbolic_func, flattened_symbolic_func)


def test_predicate():
    _check(compacted_predicate_func, flattened_predicate_func)


def test_unit_loops():
    _check(compacted_unit_loop_func, flattened_unit_loop_func)


def test_multi_alloc():
    _check(compacted_multi_alloc_func, flattened_multi_alloc_func)


def test_strided_buffer():
    _check(compacted_strided_buffer_func, flattened_strided_buffer_func)


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.FlattenBuffer()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # FlattenBuffer should do nothing on TE


if __name__ == "__main__":
    test_elementwise()
    test_gpu_workload()
    test_symbolic_shape()
    test_predicate()
    test_unit_loops()
    test_multi_alloc()
    test_strided_buffer()
    test_lower_te()
