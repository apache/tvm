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
import numpy as np
import pytest

import tvm
from tvm.script import tirx as T
from tvm.testing import env


def run_test_break_continue(func, shape, expected):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    arr_np = np.zeros(shape, dtype="int32")

    def run_and_check():
        dev = tvm.cuda(0)
        arr = tvm.runtime.tensor(arr_np, device=dev)
        mod(arr)
        dev.sync()
        np.testing.assert_allclose(arr.numpy(), expected)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_break_continue1():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([32])
        for i in T.serial(10):
            if i == 2:
                continue
            if i == 7:
                break
            A[i] = i
        # fmt: on

    expected = np.array([0, 1, 0, 3, 4, 5, 6, 0, 0, 0], dtype="int32")
    run_test_break_continue(func, (10,), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_break_continue2():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (9,), "int32")

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([32])
        idx = T.alloc_buffer((1,), "int32", scope="local")
        idx[0] = 0
        for i in T.serial(3):
            if i == 0:
                idx[0] += 1
                continue
            for j in T.serial(3):
                A[idx[0]] = i * 10 + j
                idx[0] += 1
                if j == 1:
                    break
        # fmt: on

    expected = np.array([0, 10, 11, 20, 21, 0, 0, 0, 0], dtype="int32")
    run_test_break_continue(func, (9,), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_break_continue3():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([32])
        i = T.alloc_buffer((1,), "int32", scope="local")
        i[0] = 0
        while i[0] < 10:
            if (i[0] % 2) == 1:
                i[0] += 1
                continue
            A[i[0]] = i[0]
            i[0] += 1
            if i[0] == 7:
                break
        # fmt: on

    expected = np.array([0, 0, 2, 0, 4, 0, 6, 0, 0, 0], dtype="int32")
    run_test_break_continue(func, (10,), expected)


if __name__ == "__main__":
    test_break_continue1()
    test_break_continue2()
    test_break_continue3()
