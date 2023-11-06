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
import tvm.testing
from tvm import te
import numpy as np
from tvm import topi
import re

tx = te.thread_axis("threadIdx.x")
bx = te.thread_axis("blockIdx.x")


@tvm.testing.requires_rocm
def test_hip_vectorize_load():
    num_thread = 8

    def check_hip(dtype, n, lanes):
        dev = tvm.rocm(0)
        A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i], name="B")
        s = te.create_schedule(B.op)
        block, thread = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(block, bx)
        s[B].bind(thread, tx)
        fun = tvm.build(s, [A, B], "hip", name="vector_load")
        np_a = np.random.randint(low=-128, high=127, size=(n, lanes))
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, dev)
        fun(a, b)
        tvm.testing.assert_allclose(a.numpy(), b.numpy())

    check_hip("int8", 64, 2)
    check_hip("int8", 64, 3)
    check_hip("int8", 64, 4)
    check_hip("int8", 64, 8)
    check_hip("int8", 64, 16)


@tvm.testing.requires_rocm
def test_hip_make_int8():
    def check_hip(n, value, lanes):
        dtype = "int8"
        dev = tvm.rocm(0)
        A = te.compute((n, lanes), lambda i, j: tvm.tir.const(value, dtype=dtype))
        s = te.create_schedule(A.op)
        y, x = s[A].op.axis
        s[A].vectorize(x)
        s[A].bind(y, bx)
        fun = tvm.build(s, [A], "hip", name="make_int8x4")
        np_a = np.full((n, lanes), value, dtype=dtype)
        a = tvm.nd.empty(np_a.shape, dtype, dev)
        fun(a)
        np.testing.assert_equal(a.numpy(), np_a)

    check_hip(64, np.int8(0xAB), 4)
    check_hip(64, 0, 4)
    check_hip(64, -3, 4)
    check_hip(64, np.int8(0xAB), 3)
    check_hip(64, 0, 3)
    check_hip(64, -3, 3)
    check_hip(64, np.int8(0xAB), 2)
    check_hip(64, 0, 2)
    check_hip(64, -3, 2)


@tvm.testing.requires_rocm
def test_hip_make_int4():
    def check_hip(n, value, lanes):
        dtype = "int4"
        dev = tvm.rocm(0)
        A = te.compute((n, lanes), lambda i, j: tvm.tir.const(value, dtype=dtype))
        s = te.create_schedule(A.op)
        y, x = s[A].op.axis
        s[A].vectorize(x)
        s[A].bind(y, bx)
        kernel_name = "make_int4x" + str(lanes)
        fun = tvm.build(s, [A], "hip", name=kernel_name)
        np_a = np.full((n, lanes), value, dtype="int8")
        a = tvm.nd.empty((n, lanes), dtype, dev)
        fun(a)
        np.testing.assert_equal(a.numpy(), np_a)

    check_hip(64, 1, 4)
    check_hip(64, 7, 4)
    check_hip(64, 1, 8)
    check_hip(64, 7, 8)
    check_hip(64, 1, 16)
    check_hip(64, 7, 16)
    check_hip(64, 1, 32)
    check_hip(64, 7, 32)


if __name__ == "__main__":
    tvm.testing.main()
