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
from tvm import tir
from tvm.script import ty


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


@tvm.script.tir
def compacted_elementwise_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with tir.block([]):
            tir.reads(A[i, 0:16])
            tir.writes(C[i, 0:16])
            B = tir.alloc_buffer([1, 16], "float32", scope="global")
            for j in range(0, 16):
                with tir.block() as []:
                    tir.reads(A[i, j])
                    tir.writes(B[0, j])
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with tir.block() as []:
                    tir.reads(B[0, j])
                    tir.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@tvm.script.tir
def flattened_elementwise_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")
    for i in tir.serial(0, 16):
        B_new = tir.allocate([16], "float32", "global")
        for j in tir.serial(0, 16):
            B_new[j] = tir.load("float32", A.data, ((i * 16) + j)) + 1.0
        for j in tir.serial(0, 16):
            C.data[((i * 16) + j)] = tir.load("float32", B_new, j) * 2.0


@tvm.script.tir
def compacted_gpu_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")
    for i0 in tir.thread_binding(0, 4, thread="blockIdx.x"):
        for i1 in tir.thread_binding(0, 2, thread="threadIdx.x"):
            for i2 in tir.thread_binding(0, 2, thread="vthread"):
                with tir.block([]):
                    tir.reads(A[i0 * 4 + i1 * 2 + i2, 0:16])
                    tir.writes(C[i0 * 4 + i1 * 2 + i2, 0:16])
                    B = tir.alloc_buffer([1, 16], "float32", scope="local")
                    for j in range(0, 16):
                        with tir.block() as []:
                            tir.reads(A[i0 * 4 + i1 * 2 + i2, j])
                            tir.writes(B[0, j])
                            B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
                    for j in range(0, 16):
                        with tir.block() as []:
                            tir.reads(B[0, j])
                            tir.writes(C[i0 * 4 + i1 * 2 + i2, j])
                            C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0


@tvm.script.tir
def flattened_gpu_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")

    i0 = tir.env_thread("blockIdx.x")
    i1 = tir.env_thread("threadIdx.x")
    i2 = tir.env_thread("vthread")

    tir.launch_thread(i0, 4)
    tir.launch_thread(i1, 2)
    tir.launch_thread(i2, 2)
    B = tir.allocate([16], "float32", "local")
    for j in range(0, 16):
        B[j] = tir.load("float32", A.data, i0 * 64 + i1 * 32 + i2 * 16 + j) + 1.0
    for j in range(0, 16):
        C.data[i0 * 64 + i1 * 32 + i2 * 16 + j] = tir.load("float32", B, j) * 2.0


@tvm.script.tir
def compacted_symbolic_func(a: ty.handle, c: ty.handle, n: ty.int32, m: ty.int32) -> None:
    A = tir.match_buffer(a, (n, m), "float32")
    C = tir.match_buffer(c, (n, m), "float32")

    for i in range(0, n):
        with tir.block([]):
            tir.reads(A[i, m])
            tir.writes(C[i, m])
            B = tir.alloc_buffer((m,), "float32", scope="global")
            for j in range(0, m):
                with tir.block([]) as []:
                    tir.reads(A[i, j])
                    tir.writes(B[j])
                    B[j] = A[i, j] + 1.0
            for j in range(0, m):
                with tir.block([]) as []:
                    tir.reads(B[j])
                    tir.writes(C[i, j])
                    C[i, j] = B[j] * 2.0


@tvm.script.tir
def flattened_symbolic_func(a: ty.handle, c: ty.handle, n: ty.int32, m: ty.int32) -> None:
    A = tir.match_buffer(a, (n, m), "float32")
    C = tir.match_buffer(c, (n, m), "float32")

    for i in range(0, n):
        B = tir.allocate([m], "float32", "global")
        for j in range(0, m):
            B[j] = tir.load("float32", A.data, i * m + j) + 1.0
        for j in range(0, m):
            C.data[i * m + j] = tir.load("float32", B, j) * 2.0


@tvm.script.tir
def compacted_predicate_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (32), "float32")
    C = tir.match_buffer(c, (32), "float32")

    for i, j in tir.grid(5, 7):
        with tir.block([]) as []:
            tir.reads(A[i * 7 + j])
            tir.writes(C[i * 7 + j])
            tir.where(i * 7 + j < 32)
            C[i * 7 + j] = A[i * 7 + j] + 1.0


@tvm.script.tir
def flattened_predicate_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (32), "float32")
    C = tir.match_buffer(c, (32), "float32")

    for i, j in tir.grid(5, 7):
        if i * 7 + j < 32:
            C.data[i * 7 + j] = tir.load("float32", A.data, i * 7 + j) + 1.0


@tvm.script.tir
def compacted_unit_loop_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (32), "float32")
    C = tir.match_buffer(c, (32), "float32")

    for x, y, z in tir.grid(4, 1, 8):
        with tir.block([]) as []:
            tir.reads(A[x * 8 + y * 8 + z])
            tir.writes(C[x * 8 + y * 8 + z])
            C[x * 8 + y * 8 + z] = A[x * 8 + y * 8 + z] + 1.0


@tvm.script.tir
def flattened_unit_loop_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (32), "float32")
    C = tir.match_buffer(c, (32), "float32")

    for x, z in tir.grid(4, 8):
        C.data[x * 8 + z] = tir.load("float32", A.data, x * 8 + z) + 1.0


@tvm.script.tir
def compacted_multi_alloc_func(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (32), "float32")
    D = tir.match_buffer(d, (32), "float32")

    for i in range(0, 32):
        with tir.block([]) as []:
            tir.reads(A[i])
            tir.writes(D[i])
            B = tir.alloc_buffer((32,), scope="global")
            C = tir.alloc_buffer((32,), scope="global")
            B[i] = A[i] + 1.0
            C[i] = A[i] + B[i]
            D[i] = C[i] * 2.0


@tvm.script.tir
def flattened_multi_alloc_func(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (32), "float32")
    D = tir.match_buffer(d, (32), "float32")

    for i in range(0, 32):
        B = tir.allocate((32,), "float32", "global")
        C = tir.allocate((32,), "float32", "global")
        B[i] = tir.load("float32", A.data, i) + 1.0
        C[i] = tir.load("float32", A.data, i) + tir.load("float32", B, i)
        D.data[i] = tir.load("float32", C, i) * 2.0


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


if __name__ == "__main__":
    test_elementwise()
    test_gpu_workload()
    test_symbolic_shape()
    test_predicate()
    test_unit_loops()
    test_multi_alloc()
