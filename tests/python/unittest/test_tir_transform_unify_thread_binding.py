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
from tvm import tir, te
from tvm.script import ty


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original)
    mod = tvm.tir.transform.UnifyThreadBinding()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


def _check_fail(original):
    mod = tvm.IRModule.from_expr(original)
    with pytest.raises(ValueError):
        tvm.tir.transform.UnifyThreadBinding()(mod)


@tvm.script.tir
def element_wise_thread_x(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    j1_0 = tir.env_thread("threadIdx.x")
    j0_0 = tir.env_thread("threadIdx.x")
    i = tir.env_thread("blockIdx.x")
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    tir.launch_thread(i, 128)
    with tir.launch_thread(j0_0, 4):
        for j0_1 in tir.serial(0, 32):
            tir.store(
                B.data,
                i * 128 + j0_0 * 32 + j0_1,
                tir.load("float32", A.data, i * 128 + j0_0 * 32 + j0_1) * 2.0,
                True,
            )
    tir.launch_thread(j1_0, 4)
    for j1_1 in tir.serial(0, 32):
        tir.store(
            C.data,
            i * 128 + j1_0 * 32 + j1_1,
            tir.load("float32", A.data, i * 128 + j1_0 * 32 + j1_1) + 1.0,
            True,
        )


@tvm.script.tir
def unified_element_wise_thread_x(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    thread_x = tir.env_thread("threadIdx.x")
    block_x = tir.env_thread("blockIdx.x")
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    tir.launch_thread(block_x, 128)
    with tir.launch_thread(thread_x, 4):
        for j0_1 in tir.serial(0, 32):
            tir.store(
                B.data,
                block_x * 128 + thread_x * 32 + j0_1,
                tir.load("float32", A.data, block_x * 128 + thread_x * 32 + j0_1) * 2.0,
                True,
            )
    tir.launch_thread(thread_x, 4)
    for j1_1 in tir.serial(0, 32):
        tir.store(
            C.data,
            block_x * 128 + thread_x * 32 + j1_1,
            tir.load("float32", A.data, block_x * 128 + thread_x * 32 + j1_1) + 1.0,
            True,
        )


@tvm.script.tir
def element_wise_vthread_x(a: ty.handle, b: ty.handle) -> None:
    i_0 = tir.env_thread("vthread.x")
    i_1 = tir.env_thread("threadIdx.x")
    j_0 = tir.env_thread("vthread.x")
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    tir.launch_thread(i_0, 2)
    tir.launch_thread(i_1, 64)
    tir.launch_thread(j_0, 2)
    for j_1 in tir.serial(0, 64):
        tir.store(
            B.data,
            i_0 * 8192 + i_1 * 128 + j_0 * 64 + j_1,
            tir.load("float32", A.data, i_0 * 8192 + i_1 * 128 + j_0 * 64 + j_1) * 2.0,
            True,
        )


@tvm.script.tir
def unified_element_wise_vthread_x(a: ty.handle, b: ty.handle) -> None:
    vthread_x = tir.env_thread("vthread.x")
    thread_x = tir.env_thread("threadIdx.x")
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    tir.launch_thread(vthread_x, 2)
    tir.launch_thread(thread_x, 64)
    tir.launch_thread(vthread_x, 2)
    for j_1 in tir.serial(0, 64):
        tir.store(
            B.data,
            vthread_x * 8256 + thread_x * 128 + j_1,
            tir.load("float32", A.data, vthread_x * 8256 + thread_x * 128 + j_1) * 2.0,
            True,
        )


@tvm.script.tir
def element_wise_two_thread_x_in_same_kernel_not_equal(a: ty.handle, b: ty.handle,
                                                       c: ty.handle) -> None:
    i = tir.env_thread("blockIdx.x")
    j0 = tir.env_thread("threadIdx.x")
    j1 = tir.env_thread("threadIdx.x")
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 64])
    tir.launch_thread(i, 128)
    with tir.launch_thread(j0, 128):
        tir.store(B.data, i * 64 + j0, tir.load("float32", A.data, i * 128 + j0) * 2.0, True)
    tir.launch_thread(j1, 64)
    tir.store(C.data, i * 64 + j1, tir.load("float32", A.data, i * 128 + j1) + 1.0, True)


@tvm.script.tir
def element_wise_kernels_with_different_size(a: ty.handle, b: ty.handle, c: ty.handle,
                                             d: ty.handle) -> None:
    i0 = tir.env_thread("blockIdx.x")
    j0 = tir.env_thread("threadIdx.x")
    i1 = tir.env_thread("blockIdx.x")
    j1 = tir.env_thread("threadIdx.x")
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [256, 256])
    D = tir.match_buffer(d, [256, 256])
    with tir.launch_thread(i0, 128):
        tir.launch_thread(j0, 128)
        tir.store(B.data, i0 * 128 + j0, tir.load("float32", A.data, i0 * 128 + j0) * 2.0, True)
    tir.launch_thread(i1, 256)
    tir.launch_thread(j1, 256)
    tir.store(D.data, i1 * 256 + j1, tir.load("float32", C.data, i1 * 256 + j1) + 1.0, True)


@tvm.script.tir
def unified_element_wise_kernels_with_different_size(a: ty.handle, b: ty.handle, c: ty.handle,
                                                     d: ty.handle) -> None:
    block_x = tir.env_thread("blockIdx.x")
    thread_x = tir.env_thread("threadIdx.x")
    block_x_1 = tir.env_thread("blockIdx.x")
    thread_x_1 = tir.env_thread("threadIdx.x")
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [256, 256])
    D = tir.match_buffer(d, [256, 256])
    with tir.launch_thread(block_x, 128):
        tir.launch_thread(thread_x, 128)
        tir.store(
            B.data,
            block_x * 128 + thread_x,
            tir.load("float32", A.data, block_x * 128 + thread_x) * 2.0,
            True,
        )
    tir.launch_thread(block_x_1, 256)
    tir.launch_thread(thread_x_1, 256)
    tir.store(
        D.data,
        block_x_1 * 256 + thread_x_1,
        tir.load("float32", C.data, block_x_1 * 256 + thread_x_1) + 1.0,
        True,
    )


def test_thread_x():
    _check(element_wise_thread_x, unified_element_wise_thread_x)


def test_vthread_x():
    _check(element_wise_vthread_x, unified_element_wise_vthread_x)


def test_two_thread_x_in_same_kernel_not_equal():
    _check_fail(element_wise_two_thread_x_in_same_kernel_not_equal)


def test_kernels_with_different_size():
    _check(element_wise_kernels_with_different_size,
           unified_element_wise_kernels_with_different_size)


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
