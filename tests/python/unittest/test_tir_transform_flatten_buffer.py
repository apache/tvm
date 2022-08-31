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
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


@T.prim_func
def elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in T.serial(0, 16):
        B_new_data = T.allocate([1, 16], "float32", "global")
        B_new = T.buffer_decl(shape=[1, 16], dtype="float32", data=B_new_data)
        for j in T.serial(0, 16):
            B_new[0, j] = A[i, j] + 1.0
        for j in T.serial(0, 16):
            C[i, j] = B_new[0, j] * 2.0


@T.prim_func
def flattened_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, 256, "float32")
    C = T.match_buffer(c, 256, "float32")
    T.preflattened_buffer(A, (16, 16), dtype="float32", data=A.data)
    T.preflattened_buffer(C, (16, 16), dtype="float32", data=C.data)
    for i in T.serial(0, 16):
        B_new_data = T.allocate([16], "float32", "global")
        B_new = T.buffer_decl(shape=[16], dtype="float32", data=B_new_data)
        for j in T.serial(0, 16):
            B_new[j] = A[((i * 16) + j)] + 1.0
        for j in T.serial(0, 16):
            C[((i * 16) + j)] = B_new[j] * 2.0


@T.prim_func
def gpu_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")

    i0 = T.env_thread("blockIdx.x")
    i1 = T.env_thread("threadIdx.x")
    i2 = T.env_thread("vthread")

    T.launch_thread(i0, 4)
    T.launch_thread(i1, 2)
    T.launch_thread(i2, 2)
    B_data = T.allocate([1, 16], "float32", "local")
    B = T.buffer_decl(shape=[1, 16], dtype="float32", data=B_data, scope="local")
    for j in range(0, 16):
        B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
    for j in range(0, 16):
        C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0


@T.prim_func
def flattened_gpu_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, 256, "float32")
    C = T.match_buffer(c, 256, "float32")
    T.preflattened_buffer(A, (16, 16), dtype="float32", data=A.data)
    T.preflattened_buffer(C, (16, 16), dtype="float32", data=C.data)

    i0 = T.env_thread("blockIdx.x")
    i1 = T.env_thread("threadIdx.x")
    i2 = T.env_thread("vthread")

    T.launch_thread(i0, 4)
    T.launch_thread(i1, 2)
    T.launch_thread(i2, 2)
    B_data = T.allocate([16], "float32", "local")
    B = T.buffer_decl(shape=[16], dtype="float32", data=B_data, scope="local")
    for j in range(0, 16):
        B[j] = A[i0 * 64 + i1 * 32 + i2 * 16 + j] + 1.0
    for j in range(0, 16):
        C[i0 * 64 + i1 * 32 + i2 * 16 + j] = B[j] * 2.0


@T.prim_func
def symbolic_func(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
    A = T.match_buffer(a, (n, m), "float32")
    C = T.match_buffer(c, (n, m), "float32")

    for i in range(0, n):
        B_data = T.allocate([m], "float32", "global")
        B = T.buffer_decl(shape=[m], dtype="float32", data=B_data)
        for j in range(0, m):
            B[j] = A[i, j] + 1.0
        for j in range(0, m):
            C[i, j] = B[j] * 2.0


@T.prim_func
def flattened_symbolic_func(a: T.handle, c: T.handle, n: T.int32, m: T.int32) -> None:
    A = T.match_buffer(a, n * m, "float32")
    C = T.match_buffer(c, n * m, "float32")
    T.preflattened_buffer(A, (n, m), "float32", data=A.data)
    T.preflattened_buffer(C, (n, m), "float32", data=C.data)

    for i in range(0, n):
        B_data = T.allocate([m], "float32", "global")
        B = T.buffer_decl(shape=[m], dtype="float32", data=B_data)
        for j in range(0, m):
            B[j] = A[i * m + j] + 1.0
        for j in range(0, m):
            C[i * m + j] = B[j] * 2.0


@T.prim_func
def multi_alloc_func(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (4, 32), "float32")
    D = T.match_buffer(d, (4, 32), "float32")

    for i, j in T.grid(4, 32):
        B_data = T.allocate((4, 32), "float32", scope="global")
        B = T.buffer_decl(shape=(4, 32), dtype="float32", data=B_data)
        C_data = T.allocate((4, 32), "float32", scope="global")
        C = T.buffer_decl(shape=(4, 32), dtype="float32", data=C_data)
        B[i, j] = A[i, j] + 1.0
        C[i, j] = A[i, j] + B[i, j]
        D[i, j] = C[i, j] * 2.0


@T.prim_func
def flattened_multi_alloc_func(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, 128, "float32")
    D = T.match_buffer(d, 128, "float32")
    T.preflattened_buffer(A, (4, 32), "float32", data=A.data)
    T.preflattened_buffer(D, (4, 32), "float32", data=D.data)

    for i, j in T.grid(4, 32):
        B_data = T.allocate([128], "float32", "global")
        B = T.buffer_decl(shape=[128], dtype="float32", data=B_data)
        C_data = T.allocate([128], "float32", "global")
        C = T.buffer_decl(shape=[128], dtype="float32", data=C_data)
        B[i * 32 + j] = A[i * 32 + j] + 1.0
        C[i * 32 + j] = A[i * 32 + j] + B[i * 32 + j]
        D[i * 32 + j] = C[i * 32 + j] * 2.0


@T.prim_func
def strided_buffer_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.serial(4):
        B_data = T.allocate([4, 17], "float32", "global")
        B = T.buffer_decl(shape=[4, 17], dtype="float32", data=B_data)
        B_1 = T.buffer_decl([4, 16], dtype="float32", data=B.data, strides=[17, 1])
        for i1, j in T.grid(4, 16):
            B_1[i1, j] = A[i0 * 4 + i1, j] + 1.0
        for i1, j in T.grid(4, 16):
            C[i0 * 4 + i1, j] = B_1[i1, j] * 2.0


@T.prim_func
def flattened_strided_buffer_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (256,), "float32")
    C = T.match_buffer(c, (256,), "float32")
    T.preflattened_buffer(A, [16, 16], dtype="float32", data=A.data)
    T.preflattened_buffer(C, [16, 16], dtype="float32", data=C.data)
    for i0 in T.serial(0, 4):
        B_new_data = T.allocate([68], "float32", "global")
        B_new = T.buffer_decl(shape=[68], dtype="float32", data=B_new_data)
        for i1 in T.serial(0, 4):
            for j in T.serial(0, 16):
                B_new[i1 * 17 + j] = A[i0 * 64 + i1 * 16 + j] + 1.0
        for i1 in T.serial(0, 4):
            for j in T.serial(0, 16):
                C[i0 * 64 + i1 * 16 + j] = B_new[i1 * 17 + j] * 2.0


@T.prim_func
def boolean_handling_before(a: T.Buffer[10, "bool"], b: T.Buffer[10, "bool"]) -> None:
    for i0 in T.serial(10):
        b[i0] = a[i0]


@T.prim_func
def boolean_handling_after(a: T.Buffer[10, "int8"], b: T.Buffer[10, "int8"]) -> None:
    T.preflattened_buffer(a, [10], dtype="bool", data=a.data)
    T.preflattened_buffer(b, [10], dtype="bool", data=b.data)
    # body
    for i0 in T.serial(10):
        b[i0] = T.cast(T.cast(a[i0], "bool"), "int8")


def test_elementwise():
    _check(elementwise_func, flattened_elementwise_func)


def test_gpu_workload():
    _check(gpu_func, flattened_gpu_func)


def test_symbolic_shape():
    _check(symbolic_func, flattened_symbolic_func)


def test_multi_alloc():
    _check(multi_alloc_func, flattened_multi_alloc_func)


def test_strided_buffer():
    _check(strided_buffer_func, flattened_strided_buffer_func)


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.FlattenBuffer()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # FlattenBuffer should do nothing on TE


def test_boolean_handling():
    _check(boolean_handling_before, boolean_handling_after)


if __name__ == "__main__":
    tvm.testing.main()
