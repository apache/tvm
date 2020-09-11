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
import unittest

tx = te.thread_axis("threadIdx.x")
ty = te.thread_axis("threadIdx.y")
bx = te.thread_axis("blockIdx.x")
by = te.thread_axis("blockIdx.y")


@tvm.testing.requires_rocm
def test_rocm_cross_thread_reduction():
    # based on the reduction tutorial
    n = te.size_var("n")
    m = te.size_var("m")
    A = te.placeholder((n, m), name="A")
    k = te.reduce_axis((0, m), "k")
    B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B].bind(xo, bx)
    s[B].bind(xi, ty)
    s[B].bind(s[B].op.reduce_axis[0], tx)
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    s[B].set_store_predicate(tx.var.equal(0))
    frocm = tvm.build(s, [A, B], "rocm")

    nn = 128
    ctx = tvm.rocm(0)
    a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), ctx)
    frocm(a, b)
    tvm.testing.assert_allclose(b.asnumpy(), np.sum(a.asnumpy(), axis=1), rtol=1e-4)


@tvm.testing.requires_rocm
def test_rocm_inf_nan():
    def check_inf_nan(ctx, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tx)
        fun = tvm.build(s, [A, C], "rocm")
        a = tvm.nd.empty((n,), A.dtype, ctx)
        c = tvm.nd.empty((n,), A.dtype, ctx)
        # Only need to test compiling here
        fun(a, c)

    ctx = tvm.rocm(0)

    check_inf_nan(ctx, 1, -float("inf"), "float32")
    check_inf_nan(ctx, 1, -float("inf"), "float64")
    check_inf_nan(ctx, 1, float("inf"), "float32")
    check_inf_nan(ctx, 1, float("inf"), "float64")
    check_inf_nan(ctx, 1, float("nan"), "float32")
    check_inf_nan(ctx, 1, float("nan"), "float64")


@tvm.testing.requires_rocm
def test_rocm_reduction_binding():
    k = te.reduce_axis((0, 32), "k")
    A = te.placeholder((96, 32), name="A")
    B = te.compute((96,), lambda m: te.sum(A[m, k], axis=k), name="B")
    s = te.create_schedule(B.op)

    s[B].reorder(B.op.reduce_axis[0], B.op.axis[0])

    mo, _ = s[B].split(B.op.axis[0], 32)
    s[B].bind(mo, bx)


@tvm.testing.requires_rocm
def test_rocm_copy():
    def check_rocm(dtype, n):
        A = te.placeholder((n,), name="A", dtype=dtype)
        ctx = tvm.rocm(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(a_np)
        b_np = a.asnumpy()
        tvm.testing.assert_allclose(a_np, b_np)
        tvm.testing.assert_allclose(a_np, a.asnumpy())

    for _ in range(100):
        dtype = np.random.choice(["float32", "float16", "int8", "int32"])
        logN = np.random.randint(1, 15)
        peturb = np.random.uniform(low=0.5, high=1.5)
        check_rocm(dtype, int(peturb * (2 ** logN)))


@tvm.testing.requires_rocm
def test_rocm_vectorize_add():
    num_thread = 8

    def check_rocm(dtype, n, lanes):
        A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, bx)
        s[B].bind(xi, tx)
        fun = tvm.build(s, [A, B], "rocm")
        ctx = tvm.rocm(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

    check_rocm("float32", 64, 2)
    check_rocm("float16", 64, 2)


if __name__ == "__main__":
    test_rocm_cross_thread_reduction()
    test_rocm_inf_nan()
    test_rocm_reduction_binding()
    test_rocm_copy()
    test_rocm_vectorize_add()
