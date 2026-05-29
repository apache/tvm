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

import tvm
from tvm.script import tirx as Tx


def run_test_break_continue(func, shape, expected):
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    arr_np = np.zeros(shape, dtype="int32")
    arr = tvm.runtime.tensor(arr_np, device=dev)
    mod(arr)
    np.testing.assert_allclose(arr.numpy(), expected)


def test_break_continue1():
    # fmt: off
    @Tx.prim_func
    def func(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([32])
            with Tx.thread():
                for i in Tx.serial(10):
                    if i == 2:
                        continue
                    if i == 7:
                        break
                    A[i] = i
    # fmt: on

    expected = np.array([0, 1, 0, 3, 4, 5, 6, 0, 0, 0], dtype="int32")
    run_test_break_continue(func, (10,), expected)


def test_break_continue2():
    # fmt: off
    @Tx.prim_func
    def func(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (9,), "int32")

        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([32])
            with Tx.thread():
                idx = Tx.alloc_buffer((1,), "int32", scope="local")
                idx[0] = 0
                for i in Tx.serial(3):
                    if i == 0:
                        idx[0] += 1
                        continue
                    for j in Tx.serial(3):
                        A[idx[0]] = i * 10 + j
                        idx[0] += 1
                        if j == 1:
                            break
    # fmt: on

    expected = np.array([0, 10, 11, 20, 21, 0, 0, 0, 0], dtype="int32")
    run_test_break_continue(func, (9,), expected)


def test_break_continue3():
    # fmt: off
    @Tx.prim_func
    def func(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([32])
            with Tx.thread():
                i = Tx.alloc_buffer((1,), "int32", scope="local")
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
