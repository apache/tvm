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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
