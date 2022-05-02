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
# pylint: disable=missing-function-docstring,missing-module-docstring,invalid-name,pointless-string-statement
import sys

import pytest
from tvm.ir import assert_structural_equal
from tvm.script import tir as T
from tvm.script.parser import from_source
from tvm.testing import check_error


@T.prim_func
def transformed_matmul_no_syntax_sugar(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i0, i1])
            vk = T.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            T.reads([C[vi, vj], A[vi, vk], B[vj, vk]])
            T.writes([C[vi, vj], A[vi, vk]])
            with T.init():
                C[vi, vj] = 0.0
            A[vi, vk] = A[vi, vk] + B[vj, vk]
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def transformed_matmul_syntax_sugar(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i0, i1])
            vk = T.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj], A[vi, vk])
            with T.init():
                C[vi, vj] = 0.0
            A[vi, vk] = A[vi, vk] + B[vj, vk]
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


def test_reads_writes_syntax_sugar():
    assert_structural_equal(transformed_matmul_no_syntax_sugar, transformed_matmul_syntax_sugar)


@T.prim_func
def loop_no_syntax_sugar(a: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    for i in T.serial(0, 128):
        for j in T.parallel(0, 128):
            for k in T.vectorized(0, 128):
                for x in T.unroll(0, 128):
                    for y in T.thread_binding(0, 128, thread="threadIdx.x"):
                        for z in T.thread_binding(0, 128, thread="threadIdx.x"):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


@T.prim_func
def loop_syntax_sugar(a: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    for i in T.serial(128):
        for j in T.parallel(128):
            for k in T.vectorized(128):
                for x in T.unroll(128):
                    for y in T.thread_binding(128, "threadIdx.x"):
                        for z in T.thread_binding(128, thread="threadIdx.x"):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


def loop_syntax_sugar_fail(a: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    for i in T.thread_binding(128, 128):
        A[i] = A[i] * 2.0


def test_loop_syntax_sugar():
    assert_structural_equal(loop_no_syntax_sugar, loop_syntax_sugar)


def test_syntax_sugar_fail():
    check_error(loop_syntax_sugar_fail, 3)


# match buffer - use kwargs
@T.prim_func
def elementwise_handle(
    a: T.handle,
    b: T.handle,
) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


# match buffer - use buffer with kwargs
@T.prim_func
def elementwise_buffer_kwargs(
    a: T.Buffer(shape=(128, 128, 128, 128), dtype="float32"),
    b: T.Buffer(shape=(128, 128, 128, 128), dtype="float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


# match buffer - use buffer without kwargs
@T.prim_func
def elementwise_buffer_no_kwargs(
    a: T.Buffer[(128, 128, 128, 128), "float32"],
    b: T.Buffer[(128, 128, 128, 128), "float32"],
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


def test_match_buffer_syntax_sugar():
    # with kwargs
    assert_structural_equal(elementwise_handle, elementwise_buffer_kwargs)
    # without kwargs
    assert_structural_equal(elementwise_handle, elementwise_buffer_no_kwargs)


# match buffer failed case
def test_match_buffer_no_kwargs_failed():
    with pytest.raises(ValueError) as e:

        @T.prim_func
        def elementwise_buffer_no_kwargs_failed(
            a: T.Buffer[(128, 128, 128, 128)],
            b: T.Buffer[(128, 128, 128, 128)],
        ) -> None:
            pass


# dynamic shape gemm
@T.prim_func
def gemm_dyn_shape(a: T.handle, b: T.handle, c: T.handle):
    N = T.var("int32")
    M = T.var("int32")
    K = T.var("int32")
    A = T.match_buffer(a, (N, K), "float32")
    B = T.match_buffer(b, (K, M), "float32")
    C = T.match_buffer(c, (N, M), "float32")
    for i, j, k in T.grid(N, M, K):
        with T.block("gemm"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dynamic_shape_gemm():
    gemm_dyn_shape_roundtrip = from_source(gemm_dyn_shape.script())
    assert_structural_equal(gemm_dyn_shape, gemm_dyn_shape_roundtrip)


@T.prim_func
def preflattened_buffer_map(A: T.handle, B: T.handle):
    A_1 = T.match_buffer(A, [1])
    T.preflattened_buffer(A_1, [1], align=T.int32(1), offset_factor=T.int64(2))
    B_1 = T.match_buffer(B, [1])
    T.preflattened_buffer(B_1, [1])
    B_1[0] = A_1[0]


def test_preflattened_buffer_map():
    A_var = [
        k for k, _ in preflattened_buffer_map.preflattened_buffer_map.items() if k.name == "A"
    ][0]
    assert preflattened_buffer_map.preflattened_buffer_map[A_var].data_alignment == 1
    assert preflattened_buffer_map.preflattened_buffer_map[A_var].offset_factor == 2


@T.prim_func
def match_buffer_int64(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (T.int64(128), T.int64(128)), dtype="float32")
    B = T.alloc_buffer((T.int64(128), T.int64(128)), dtype="float32")
    C = T.match_buffer(c, (T.int64(128), T.int64(128)), dtype="float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def match_buffer_int64_after_roundtrip(
    A: T.Buffer[(T.int64(128), T.int64(128)), "float32"],
    C: T.Buffer[(T.int64(128), T.int64(128)), "float32"],
) -> None:
    B = T.alloc_buffer((T.int64(128), T.int64(128)), dtype="float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


def test_match_buffer_int64():
    original = match_buffer_int64
    after_roundtrip = match_buffer_int64_after_roundtrip
    assert_structural_equal(original, after_roundtrip, True)


def test_letstmt_bufferload_without_type_annotation():
    # Variable assignment of PrimExpr types uses the dtype of the
    # PrimExpr to determine the variable's dtype.  Parsing of
    # buf[indices] is done by generating a BufferSlice object, which
    # handles both store and load cases.  BufferSlice is not a
    # PrimExpr, and implements BufferSlice.dtype explicitly.

    # Failure occurred during parsing of the tvmscript.
    @T.prim_func
    def func_without_type_annotation(A: T.Buffer[(1,), "int32"]):
        x = A[0]
        T.evaluate(x)


def test_letstmt_bind_with_constant():
    @T.prim_func
    def constant_binds():
        x = 1
        y = 42.0
        T.evaluate(T.cast(x, "float32") + y)

    @T.prim_func
    def constant_binds_wrapped():
        x = T.int32(1)
        y = T.float32(42.0)
        T.evaluate(T.cast(x, "float32") + y)

    assert_structural_equal(constant_binds, constant_binds_wrapped)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
