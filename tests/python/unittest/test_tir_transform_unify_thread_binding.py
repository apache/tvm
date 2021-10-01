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
import tvm
from tvm import te
from tvm.script import tir as T


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original)
    mod = tvm.tir.transform.UnifyThreadBinding()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


def _check_fail(original):
    mod = tvm.IRModule.from_expr(original)
    with pytest.raises(ValueError):
        tvm.tir.transform.UnifyThreadBinding()(mod)


@T.prim_func
def element_wise_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    j1_0 = T.env_thread("threadIdx.x")
    j0_0 = T.env_thread("threadIdx.x")
    i = T.env_thread("blockIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    T.launch_thread(i, 128)
    with T.launch_thread(j0_0, 4):
        for j0_1 in T.serial(0, 32):
            T.store(
                B.data,
                i * 128 + j0_0 * 32 + j0_1,
                T.load("float32", A.data, i * 128 + j0_0 * 32 + j0_1) * 2.0,
                True,
            )
    T.launch_thread(j1_0, 4)
    for j1_1 in T.serial(0, 32):
        T.store(
            C.data,
            i * 128 + j1_0 * 32 + j1_1,
            T.load("float32", A.data, i * 128 + j1_0 * 32 + j1_1) + 1.0,
            True,
        )


@T.prim_func
def unified_element_wise_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    thread_x = T.env_thread("threadIdx.x")
    block_x = T.env_thread("blockIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    T.launch_thread(block_x, 128)
    with T.launch_thread(thread_x, 4):
        for j0_1 in T.serial(0, 32):
            T.store(
                B.data,
                block_x * 128 + thread_x * 32 + j0_1,
                T.load("float32", A.data, block_x * 128 + thread_x * 32 + j0_1) * 2.0,
                True,
            )
    T.launch_thread(thread_x, 4)
    for j1_1 in T.serial(0, 32):
        T.store(
            C.data,
            block_x * 128 + thread_x * 32 + j1_1,
            T.load("float32", A.data, block_x * 128 + thread_x * 32 + j1_1) + 1.0,
            True,
        )


@T.prim_func
def element_wise_vthread_x(a: T.handle, b: T.handle) -> None:
    i_0 = T.env_thread("vthread.x")
    i_1 = T.env_thread("threadIdx.x")
    j_0 = T.env_thread("vthread.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    T.launch_thread(i_0, 2)
    T.launch_thread(i_1, 64)
    T.launch_thread(j_0, 2)
    for j_1 in T.serial(0, 64):
        T.store(
            B.data,
            i_0 * 8192 + i_1 * 128 + j_0 * 64 + j_1,
            T.load("float32", A.data, i_0 * 8192 + i_1 * 128 + j_0 * 64 + j_1) * 2.0,
            True,
        )


@T.prim_func
def unified_element_wise_vthread_x(a: T.handle, b: T.handle) -> None:
    vthread_x = T.env_thread("vthread.x")
    thread_x = T.env_thread("threadIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    T.launch_thread(vthread_x, 2)
    T.launch_thread(thread_x, 64)
    T.launch_thread(vthread_x, 2)
    for j_1 in T.serial(0, 64):
        T.store(
            B.data,
            vthread_x * 8256 + thread_x * 128 + j_1,
            T.load("float32", A.data, vthread_x * 8256 + thread_x * 128 + j_1) * 2.0,
            True,
        )


@T.prim_func
def element_wise_two_thread_x_in_same_kernel_not_equal(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    i = T.env_thread("blockIdx.x")
    j0 = T.env_thread("threadIdx.x")
    j1 = T.env_thread("threadIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 64])
    T.launch_thread(i, 128)
    with T.launch_thread(j0, 128):
        T.store(B.data, i * 64 + j0, T.load("float32", A.data, i * 128 + j0) * 2.0, True)
    T.launch_thread(j1, 64)
    T.store(C.data, i * 64 + j1, T.load("float32", A.data, i * 128 + j1) + 1.0, True)


@T.prim_func
def element_wise_kernels_with_different_size(
    a: T.handle, b: T.handle, c: T.handle, d: T.handle
) -> None:
    i0 = T.env_thread("blockIdx.x")
    j0 = T.env_thread("threadIdx.x")
    i1 = T.env_thread("blockIdx.x")
    j1 = T.env_thread("threadIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [256, 256])
    D = T.match_buffer(d, [256, 256])
    with T.launch_thread(i0, 128):
        T.launch_thread(j0, 128)
        T.store(B.data, i0 * 128 + j0, T.load("float32", A.data, i0 * 128 + j0) * 2.0, True)
    T.launch_thread(i1, 256)
    T.launch_thread(j1, 256)
    T.store(D.data, i1 * 256 + j1, T.load("float32", C.data, i1 * 256 + j1) + 1.0, True)


@T.prim_func
def unified_element_wise_kernels_with_different_size(
    a: T.handle, b: T.handle, c: T.handle, d: T.handle
) -> None:
    block_x = T.env_thread("blockIdx.x")
    thread_x = T.env_thread("threadIdx.x")
    block_x_1 = T.env_thread("blockIdx.x")
    thread_x_1 = T.env_thread("threadIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [256, 256])
    D = T.match_buffer(d, [256, 256])
    with T.launch_thread(block_x, 128):
        T.launch_thread(thread_x, 128)
        T.store(
            B.data,
            block_x * 128 + thread_x,
            T.load("float32", A.data, block_x * 128 + thread_x) * 2.0,
            True,
        )
    T.launch_thread(block_x_1, 256)
    T.launch_thread(thread_x_1, 256)
    T.store(
        D.data,
        block_x_1 * 256 + thread_x_1,
        T.load("float32", C.data, block_x_1 * 256 + thread_x_1) + 1.0,
        True,
    )


def test_thread_x():
    _check(element_wise_thread_x, unified_element_wise_thread_x)


def test_vthread_x():
    _check(element_wise_vthread_x, unified_element_wise_vthread_x)


def test_two_thread_x_in_same_kernel_not_equal():
    _check_fail(element_wise_two_thread_x_in_same_kernel_not_equal)


def test_kernels_with_different_size():
    _check(
        element_wise_kernels_with_different_size, unified_element_wise_kernels_with_different_size
    )


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
    test_thread_x()
    test_vthread_x()
    test_two_thread_x_in_same_kernel_not_equal()
    test_kernels_with_different_size()
    test_lower_te()
