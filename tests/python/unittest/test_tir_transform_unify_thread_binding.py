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
import pytest
import sys

import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.UnifyThreadBinding()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(
        mod["main"], transformed.with_attr("global_symbol", "main"), True
    )


def _check_fail(original):
    mod = tvm.IRModule.from_expr(original)
    with pytest.raises(ValueError):
        tvm.tir.transform.UnifyThreadBinding()(mod)


@T.prim_func
def element_wise_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i in T.thread_binding(0, 128, "blockIdx.x"):
        for j0_0 in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.thread_binding(0, 4, "threadIdx.x"):
            for j1_1 in T.serial(0, 32):
                with T.block(""):
                    C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@T.prim_func
def unified_element_wise_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for blockIdx_x in T.thread_binding(0, 128, "blockIdx.x"):
        for threadIdx_x in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                        A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                    )
            for j1_1 in T.serial(0, 32):
                with T.block(""):
                    C[blockIdx_x, threadIdx_x * 32 + j1_1] = (
                        B[blockIdx_x, threadIdx_x * 32 + j1_1] + 1.0
                    )


@T.prim_func
def element_wise_thread_x_different_dtype(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for i in T.thread_binding(128, "blockIdx.x"):
        for j0_0 in T.thread_binding(4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.thread_binding(T.int64(4), "threadIdx.x"):
            for j1_1 in T.serial(T.int64(32)):
                with T.block(""):
                    C[i, j1_0 * T.int64(32) + j1_1] = B[i, j1_0 * T.int64(32) + j1_1] + 1.0


@T.prim_func
def unified_element_wise_thread_x_different_dtype(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for blockIdx_x in T.thread_binding(128, "blockIdx.x"):
        for threadIdx_x in T.thread_binding(4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                        A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                    )
            for j1_1 in T.serial(T.int64(32)):
                with T.block(""):
                    C[blockIdx_x, T.cast(threadIdx_x, "int64") * T.int64(32) + j1_1] = (
                        B[blockIdx_x, T.cast(threadIdx_x, "int64") * T.int64(32) + j1_1] + 1.0
                    )


@T.prim_func
def element_wise_env_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    j1_0 = T.env_thread("threadIdx.x")
    j0_0 = T.env_thread("threadIdx.x")
    i = T.env_thread("blockIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    T.launch_thread(i, 128)
    T.launch_thread(j0_0, 4)
    T.launch_thread(j1_0, 4)

    for j0_1 in T.serial(0, 32):
        with T.block(""):
            B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
    for j1_1 in T.serial(0, 32):
        with T.block(""):
            C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@T.prim_func
def unified_element_wise_env_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for blockIdx_x in T.thread_binding(0, 128, "blockIdx.x"):
        for threadIdx_x in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                        A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                    )
            for j1_1 in T.serial(0, 32):
                with T.block(""):
                    C[blockIdx_x, threadIdx_x * 32 + j1_1] = (
                        B[blockIdx_x, threadIdx_x * 32 + j1_1] + 1.0
                    )


@T.prim_func
def element_wise_vthread_x(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    for i_0 in T.thread_binding(0, 2, "vthread.x"):
        for i_1 in T.thread_binding(0, 64, "threadIdx.x"):
            for j_0 in T.thread_binding(0, 2, "vthread.x"):
                for j_1 in T.serial(0, 64):
                    with T.block(""):
                        B[i_0 * 64 + i_1, j_0 * 64 + j_1] = A[i_0 * 64 + i_1, j_0 * 64 + j_1] * 2.0


@T.prim_func
def unified_element_wise_vthread_x(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    for vthread_x in T.thread_binding(0, 2, "vthread.x"):
        for threadIdx_x in T.thread_binding(0, 64, "threadIdx.x"):
            for j_1 in T.serial(0, 64):
                with T.block(""):
                    B[vthread_x * 64 + threadIdx_x, vthread_x * 64 + j_1] = (
                        A[vthread_x * 64 + threadIdx_x, vthread_x * 64 + j_1] * 2.0
                    )


@T.prim_func
def element_wise_two_thread_x_in_same_kernel_not_equal(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 64])
    for i in T.thread_binding(0, 128, "blockIdx.x"):
        for j0 in T.thread_binding(0, 128, "threadIdx.x"):
            B[i, j0] = A[i, j0] * 2.0
        for j1 in T.thread_binding(0, 64, "threadIdx.x"):
            C[i, j1] = A[i, j1] + 1.0


@T.prim_func
def element_wise_kernels_with_different_size(
    a: T.handle, b: T.handle, c: T.handle, d: T.handle
) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [256, 256])
    D = T.match_buffer(d, [256, 256])
    for i0 in T.thread_binding(0, 128, "blockIdx.x"):
        for j0 in T.thread_binding(0, 128, "threadIdx.x"):
            B[i0, j0] = A[i0, j0] * 2.0
    for i1 in T.thread_binding(0, 256, "blockIdx.x"):
        for j1 in T.thread_binding(0, 256, "threadIdx.x"):
            D[i1, j1] = C[i1, j1] + 1.0


@T.prim_func
def unified_element_wise_kernels_with_different_size(
    a: T.handle, b: T.handle, c: T.handle, d: T.handle
) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [256, 256])
    D = T.match_buffer(d, [256, 256])
    for blockIdx_x in T.thread_binding(0, 128, "blockIdx.x"):
        for threadIdx_x in T.thread_binding(0, 128, "threadIdx.x"):
            B[blockIdx_x, threadIdx_x] = A[blockIdx_x, threadIdx_x] * 2.0
    for blockIdx_x in T.thread_binding(0, 256, "blockIdx.x"):
        for threadIdx_x in T.thread_binding(0, 256, "threadIdx.x"):
            D[blockIdx_x, threadIdx_x] = C[blockIdx_x, threadIdx_x] + 1.0


@T.prim_func
def element_wise_implicit_block(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i in T.thread_binding(0, 128, "threadIdx.y"):
        for j0_0 in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.thread_binding(0, 4, "threadIdx.x"):
            for j1_1 in T.serial(0, 32):
                with T.block(""):
                    C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@T.prim_func
def unified_element_wise_implicit_block(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for blockIdx_x in T.thread_binding(0, 128, "threadIdx.y"):
        for threadIdx_x in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                        A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                    )
            for j1_1 in T.serial(0, 32):
                with T.block(""):
                    C[blockIdx_x, threadIdx_x * 32 + j1_1] = (
                        B[blockIdx_x, threadIdx_x * 32 + j1_1] + 1.0
                    )


def test_thread_x():
    _check(element_wise_thread_x, unified_element_wise_thread_x)


def test_thread_x_different_dtype():
    _check(element_wise_thread_x_different_dtype, unified_element_wise_thread_x_different_dtype)


def test_env_thread_x():
    _check(element_wise_env_thread_x, unified_element_wise_env_thread_x)


def test_vthread_x():
    _check(element_wise_vthread_x, unified_element_wise_vthread_x)


def test_two_thread_x_in_same_kernel_not_equal():
    _check_fail(element_wise_two_thread_x_in_same_kernel_not_equal)


def test_kernels_with_different_size():
    _check(
        element_wise_kernels_with_different_size, unified_element_wise_kernels_with_different_size
    )


def test_implicit_block():
    _check(element_wise_implicit_block, unified_element_wise_implicit_block)


def test_inner_binding_with_annotation():
    @T.prim_func
    def inner_binding_with_annotation(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        for bx in T.thread_binding(32, "blockIdx.x"):
            for tx in T.thread_binding(2, "threadIdx.x", annotations={"my_annotation": 1}):
                with T.block("block"):
                    v = T.axis.spatial(64, bx * 2 + tx)
                    B[v] = A[v]

    @T.prim_func
    def unified_inner_binding_with_annotation(
        A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")
    ):
        for blockIdx_x in T.thread_binding(32, thread="blockIdx.x"):
            for threadIdx_x in T.thread_binding(2, thread="threadIdx.x"):
                for var in T.serial(1, annotations={"my_annotation": 1}):
                    with T.block("block"):
                        v = T.axis.spatial(64, blockIdx_x * 2 + threadIdx_x)
                        T.reads(A[v])
                        T.writes(B[v])
                        B[v] = A[v]

    _check(inner_binding_with_annotation, unified_inner_binding_with_annotation)


def test_lower_te():
    a = te.placeholder((32, 2, 2))
    b = te.compute((32, 2, 2), lambda i, j, k: a[i, j, k] * 2.0)
    s = te.create_schedule(b.op)
    s[b].bind(b.op.axis[1], te.thread_axis("threadIdx.x"))
    s[b].bind(b.op.axis[2], te.thread_axis("threadIdx.x"))
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [a, b])
    mod = tvm.tir.transform.UnifyThreadBinding()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # UnifyThreadBinding should do nothing on TE


if __name__ == "__main__":
    tvm.testing.main()
