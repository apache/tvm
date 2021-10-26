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
from tvm import te
from tvm.script import tir as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed)


@T.prim_func
def element_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16))
    C = T.match_buffer(c, (16, 16))
    B = T.alloc_buffer((16, 16))
    for i0 in range(0, 16):
        for j0 in range(0, 16):
            with T.block():
                i, j = T.axis.remap("SS", [i0, j0])
                B[i, j] = A[i, j] + 1.0
        for j0 in range(0, 16):
            with T.block():
                i, j = T.axis.remap("SS", [i0, j0])
                C[i, j] = B[i, j] * 2.0


@T.prim_func
def transformed_element_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 16])
    C = T.match_buffer(c, [16, 16])

    for i_0 in range(0, 16):
        with T.block():
            T.reads([A[i_0, 0:16]])
            T.writes([C[i_0, 0:16]])
            B = T.alloc_buffer([16, 16])
            for j_0 in T.serial(0, 16):
                with T.block():
                    i, j = T.axis.remap("SS", [i_0, j_0])
                    B[i, j] = A[i, j] + 1.0
            for j_0 in T.serial(0, 16):
                with T.block():
                    i, j = T.axis.remap("SS", [i_0, j_0])
                    C[i, j] = B[i, j] * 2.0


@T.prim_func
def original_func() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    for i0, j0 in T.grid(128, 128):
        with T.block():
            i, j = T.axis.remap("SS", [i0, j0])
            A[i, j] = T.float32(0)
    for i0, j0, k0 in T.grid(32, 32, 32):
        with T.block():
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            B = T.alloc_buffer((128, 128), "float32")
            C = T.alloc_buffer((128, 128), "float32")
            D = T.alloc_buffer((128, 128), "float32")
            if k == 0:
                for ii, jj in T.grid(4, 4):
                    B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
            for ii, jj in T.grid(4, 4):
                for kk in range(0, 4):
                    B[i * 4 + ii, j * 4 + jj] += C[i * 4 + ii, k * 4 + kk]
                for kk in range(0, 4):
                    B[i * 4 + ii, j * 4 + jj] += (
                        D[j * 4 + jj, k * 4 + kk] * C[i * 4 + ii, k * 4 + kk]
                    )


@T.prim_func
def transformed_func() -> None:
    A = T.alloc_buffer([128, 128])
    for i0, j0 in T.grid(128, 128):
        with T.block():
            i, j = T.axis.remap("SS", [i0, j0])
            A[i, j] = T.float32(0)
    for i0, j0, k0 in T.grid(32, 32, 32):
        with T.block():
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            B = T.alloc_buffer([128, 128])
            if k == 0:
                for ii, jj in T.grid(4, 4):
                    B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
            for ii, jj in T.grid(4, 4):
                with T.block(""):
                    T.reads([B[((i * 4) + ii), ((j * 4) + jj)]])
                    T.writes([B[((i * 4) + ii), ((j * 4) + jj)]])
                    C = T.alloc_buffer([128, 128])
                    for kk in T.serial(0, 4):
                        B[((i * 4) + ii), ((j * 4) + jj)] = (
                            B[((i * 4) + ii), ((j * 4) + jj)] + C[((i * 4) + ii), ((k * 4) + kk)]
                        )
                    for kk in T.serial(0, 4):
                        with T.block(""):
                            T.reads(
                                [
                                    B[((i * 4) + ii), ((j * 4) + jj)],
                                    C[((i * 4) + ii), ((k * 4) + kk)],
                                ]
                            )
                            T.writes([B[((i * 4) + ii), ((j * 4) + jj)]])
                            D = T.alloc_buffer([128, 128])
                            B[((i * 4) + ii), ((j * 4) + jj)] = B[
                                ((i * 4) + ii), ((j * 4) + jj)
                            ] + (
                                D[((j * 4) + jj), ((k * 4) + kk)]
                                * C[((i * 4) + ii), ((k * 4) + kk)]
                            )


@T.prim_func
def match_buffer_func() -> None:
    C = T.alloc_buffer((128, 128))
    for i in range(128):
        with T.block():
            vi = T.axis.S(128, i)
            C0 = T.match_buffer(C[vi, 0:128], (128))
            for j in range(128):
                with T.block():
                    jj = T.axis.S(128, j)
                    C1 = T.match_buffer(C0[jj], ())
                    C1[()] = 0


@T.prim_func
def transformed_match_buffer_func() -> None:
    for i in range(0, 128):
        with T.block():
            vi = T.axis.S(128, i)
            C = T.alloc_buffer((128, 128))
            C0 = T.match_buffer(C[vi, 0:128], (128))
            for j in range(128):
                with T.block():
                    jj = T.axis.S(128, j)
                    C1 = T.match_buffer(C0[jj], ())
                    C1[()] = 0


@T.prim_func
def opaque_access(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [1024])
    B = T.match_buffer(b, [1024])
    A_cache = T.alloc_buffer([1024])
    for i in T.serial(0, 8):
        with T.block():
            vi = T.axis.S(8, i)
            with T.block():
                v = T.axis.S(8, vi)
                T.reads([A[(v * 128) : ((v * 128) + 128)]])
                T.writes([A_cache[(v * 128) : ((v * 128) + 128)]])
                T.evaluate(
                    T.call_extern(
                        "test",
                        A_cache.data,
                        (v * 128),
                        128,
                        A.data,
                        (v * 128),
                        128,
                        dtype="float32",
                    )
                )
            for j in T.serial(0, 128):
                with T.block():
                    v = T.axis.S(1024, vi * 128 + j)
                    T.reads([A_cache[v]])
                    T.writes([B[v]])
                    B[v] = A_cache[v]


@T.prim_func
def transformed_opaque_access(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [1024])
    B = T.match_buffer(b, [1024])
    for i in T.serial(0, 8):
        with T.block():
            vi = T.axis.S(8, i)
            T.reads(A[vi * 128 : vi * 128 + 128])
            T.writes(B[vi * 128 : vi * 128 + 128])
            A_cache = T.alloc_buffer([1024])
            with T.block():
                v = T.axis.S(8, vi)
                T.reads([A[v * 128 : v * 128 + 128]])
                T.writes([A_cache[v * 128 : v * 128 + 128]])
                T.evaluate(
                    T.call_extern(
                        "test", A_cache.data, v * 128, 128, A.data, v * 128, 128, dtype="float32"
                    )
                )
            for j in T.serial(0, 128):
                with T.block():
                    v = T.axis.S(1024, vi * 128 + j)
                    T.reads([A_cache[v]])
                    T.writes([B[v]])
                    B[v] = A_cache[v]


def test_elementwise():
    _check(element_func, transformed_element_func)


def test_locate_buffer_allocation():
    _check(original_func, transformed_func)


def test_match_buffer_allocation():
    _check(match_buffer_func, transformed_match_buffer_func)


def test_opaque_access():
    _check(opaque_access, transformed_opaque_access)


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(orig_mod)
    tvm.ir.assert_structural_equal(
        mod, orig_mod
    )  # PlanAndUpdateBufferAllocationLocation should do nothing on TE


if __name__ == "__main__":
    test_elementwise()
    test_locate_buffer_allocation()
    test_match_buffer_allocation()
    test_opaque_access()
    test_lower_te()
