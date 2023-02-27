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
from tvm import te
import tvm.testing
from tvm.script import tir


def test_floor_div_op():
    target = "llvm"
    dev = tvm.device(target)
    N = 100
    divisor = 5

    @tir.prim_func
    def func_64(
        A: tir.Buffer((N + 100, 2), "int64"),
        B: tir.Buffer((N), "int64"),
        C: tir.Buffer((N), "int64"),
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int64"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int64"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    @tir.prim_func
    def func_32(
        A: tir.Buffer((N + 100, 2), "int32"),
        B: tir.Buffer((N), "int32"),
        C: tir.Buffer((N), "int32"),
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int32"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int32"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    @tir.prim_func
    def func_16(
        A: tir.Buffer((N + 100, 2), "int16"),
        B: tir.Buffer((N), "int16"),
        C: tir.Buffer((N), "int16"),
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int16"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int16"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    @tir.prim_func
    def func_8(
        A: tir.Buffer((N + 100, 2), "int8"), B: tir.Buffer((N), "int8"), C: tir.Buffer((N), "int8")
    ):
        for i in tir.serial(N):
            with tir.block("A"):
                v_i = tir.axis.spatial(N, i)
                A[v_i, 0] = tir.floordiv(C[v_i] - tir.max_value("int8"), divisor)
                A[v_i, 1] = tir.floormod(C[v_i] - tir.max_value("int8"), divisor)
                A[v_i + 100, 0] = tir.floordiv(B[v_i], divisor)
                A[v_i + 100, 1] = tir.floormod(B[v_i], divisor)

    for opfunc, type in [
        (func_8, "int8"),
        (func_16, "int16"),
        (func_32, "int32"),
        (func_64, "int64"),
    ]:
        built = tvm.build(opfunc, target=target)
        x_data = np.random.randint(te.min_value(type), te.max_value(type), size=(100), dtype=type)
        y_data = np.asarray([i for i in range(N)], dtype=type)

        a_dev = tvm.nd.empty([N + 100, 2], type, dev)
        b_dev = tvm.nd.array(x_data, dev)
        c_dev = tvm.nd.array(y_data, dev)

        built(a_dev, b_dev, c_dev)

        a = a_dev.numpy()
        b = b_dev.numpy()
        c = c_dev.numpy()

        # python modulo behaves a bit different to tvm floormod for negative numbers
        for i in range(N + 100):
            if a[i, 1] < 0:
                a[i, 1] = divisor + a[i, 1]

        np.testing.assert_array_equal(a[:100, 0], (c - te.max_value(type)) // divisor)
        np.testing.assert_array_equal(a[:100, 1], (c - te.max_value(type)) % divisor)
        np.testing.assert_array_equal(a[100 : N + 100, 0], b // divisor)
        np.testing.assert_array_equal(a[100 : N + 100, 1], b % divisor)


if __name__ == "__main__":
    tvm.testing.main()
