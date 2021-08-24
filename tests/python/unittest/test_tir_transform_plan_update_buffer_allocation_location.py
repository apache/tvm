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
from tvm.script import ty


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    print(tvm.script.asscript(original))
    print(tvm.script.asscript(mod["main"]))
    print(tvm.script.asscript(transformed))
    tvm.ir.assert_structural_equal(mod["main"], transformed)


@tvm.script.tir
def element_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16))
    C = tir.match_buffer(c, (16, 16))
    B = tir.alloc_buffer((16, 16))
    for i_0 in range(0, 16):
        for j_0 in range(0, 16):
            with tir.block([16, 16]) as [i, j]:
                B[i, j] = A[i, j] + 1.0
        for j_0 in range(0, 16):
            with tir.block([16, 16]) as [i, j]:
                C[i, j] = B[i, j] * 2.0


@tvm.script.tir
def transformed_element_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16])
    C = tir.match_buffer(c, [16, 16])

    for i_0 in range(0, 16):
        with tir.block([]):
            tir.reads([A[i_0, 0:16]])
            tir.writes([C[i_0, 0:16]])
            B = tir.alloc_buffer([16, 16])
            for j_0 in tir.serial(0, 16):
                with tir.block([16, 16], "") as [i, j]:
                    tir.bind(i, i_0)
                    tir.bind(j, j_0)
                    B[i, j] = A[i, j] + 1.0
            for j_0 in tir.serial(0, 16):
                with tir.block([16, 16], "") as [i, j]:
                    tir.bind(i, i_0)
                    tir.bind(j, j_0)
                    C[i, j] = B[i, j] * 2.0


@tvm.script.tir
def original_func() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([128, 128]) as [i, j]:
        A[i, j] = tir.float32(0)
    with tir.block([32, 32, tir.reduce_axis(0, 32)]) as [i, j, k]:
        B = tir.alloc_buffer((128, 128), "float32")
        C = tir.alloc_buffer((128, 128), "float32")
        D = tir.alloc_buffer((128, 128), "float32")
        if k == 0:
            for ii, jj in tir.grid(4, 4):
                B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
        for ii, jj in tir.grid(4, 4):
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += C[i * 4 + ii, k * 4 + kk]
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += D[j * 4 + jj, k * 4 + kk] * C[i * 4 + ii, k * 4 + kk]


@tvm.script.tir
def transformed_func() -> None:
    A = tir.alloc_buffer([128, 128])
    with tir.block([128, 128], "") as [i, j]:
        A[i, j] = tir.float32(0)
    with tir.block([32, 32, tir.reduce_axis(0, 32)], "") as [i, j, k]:
        B = tir.alloc_buffer([128, 128])
        if k == 0:
            for ii, jj in tir.grid(4, 4):
                B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
        for ii, jj in tir.grid(4, 4):
            with tir.block([], ""):
                tir.reads([B[((i * 4) + ii), ((j * 4) + jj)]])
                tir.writes([B[((i * 4) + ii), ((j * 4) + jj)]])
                C = tir.alloc_buffer([128, 128])
                for kk in tir.serial(0, 4):
                    B[((i * 4) + ii), ((j * 4) + jj)] = (
                        B[((i * 4) + ii), ((j * 4) + jj)] + C[((i * 4) + ii), ((k * 4) + kk)]
                    )
                for kk in tir.serial(0, 4):
                    with tir.block([], ""):
                        tir.reads(
                            [
                                B[((i * 4) + ii), ((j * 4) + jj)],
                                C[((i * 4) + ii), ((k * 4) + kk)],
                            ]
                        )
                        tir.writes([B[((i * 4) + ii), ((j * 4) + jj)]])
                        D = tir.alloc_buffer([128, 128])
                        B[((i * 4) + ii), ((j * 4) + jj)] = B[((i * 4) + ii), ((j * 4) + jj)] + (
                            D[((j * 4) + jj), ((k * 4) + kk)] * C[((i * 4) + ii), ((k * 4) + kk)]
                        )


@tvm.script.tir
def match_buffer_func() -> None:
    C = tir.alloc_buffer((128, 128))
    with tir.block([128]) as [vi]:
        C0 = tir.match_buffer(C[vi, 0:128], (128))
        with tir.block([128]) as [jj]:
            C1 = tir.match_buffer(C0[jj], ())
            C1[()] = 0


@tvm.script.tir
def transformed_match_buffer_func() -> None:
    for i in range(0, 128):
        with tir.block([128]) as [vi]:
            tir.bind(vi, i)
            C = tir.alloc_buffer((128, 128))
            C0 = tir.match_buffer(C[vi, 0:128], (128))
            with tir.block([128]) as [jj]:
                C1 = tir.match_buffer(C0[jj], ())
                C1[()] = 0


@tvm.script.tir
def opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [1024])
    B = tir.match_buffer(b, [1024])
    A_cache = tir.alloc_buffer([1024])
    for i in tir.serial(0, 8):
        with tir.block([8]) as [vi]:
            with tir.block([8]) as [v]:
                tir.bind(v, vi)
                tir.reads([A[(v * 128) : ((v * 128) + 128)]])
                tir.writes([A_cache[(v * 128) : ((v * 128) + 128)]])
                tir.evaluate(
                    tir.call_extern(
                        "test", A_cache.data, (v * 128), 128, A.data, (v * 128), 128, dtype="float32"
                    )
                )
            for j in tir.serial(0, 128):
                with tir.block([1024]) as [v]:
                    tir.bind(v, ((vi * 128) + j))
                    tir.reads([A_cache[v]])
                    tir.writes([B[v]])
                    B[v] = A_cache[v]


@tvm.script.tir
def transformed_opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [1024])
    B = tir.match_buffer(b, [1024])
    for i in tir.serial(0, 8):
        with tir.block([8]) as [vi]:
            tir.reads(A[vi * 128 : vi * 128 + 128])
            tir.writes(B[vi * 128 : vi * 128 + 128])
            A_cache = tir.alloc_buffer([1024])
            with tir.block([8]) as [v]:
                tir.bind(v, vi)
                tir.reads([A[v * 128 : v * 128 + 128]])
                tir.writes([A_cache[v * 128 : v * 128 + 128]])
                tir.evaluate(
                    tir.call_extern(
                        "test", A_cache.data, v * 128, 128, A.data, v * 128, 128, dtype="float32"
                    )
                )
            for j in tir.serial(0, 128):
                with tir.block([1024]) as [v]:
                    tir.bind(v, ((vi * 128) + j))
                    tir.reads([A_cache[v]])
                    tir.writes([B[v]])
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
