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
from tvm import te
import numpy as np


def test_trace_default_action():
    n = 2
    x = te.placeholder((n, n, n), name="X", dtype="float32")
    y = te.compute(x.shape, lambda i, j, k: tvm.tir.trace([i, j, k, x[i][j][k]]))
    s = te.create_schedule(y.op)
    f = tvm.build(s, [x, y], target="llvm")
    xnd = tvm.nd.array(np.ones((n, n, n), dtype=x.dtype))
    ynd = tvm.nd.array(np.zeros((n, n, n), dtype=y.dtype))
    f(xnd, ynd)


def test_trace_expr_assign():
    @tvm.register_func("tvm.tir.trace_callback2")
    def trace_buffer(x):
        return

    def check_assign(dtype):
        n = 4
        x = te.placeholder((n, n, n), name="X", dtype=dtype)
        y = te.compute(
            x.shape, lambda i, j, k: tvm.tir.trace([x[i][j][k]], "tvm.tir.trace_callback2")
        )
        z = te.compute(
            x.shape, lambda i, j, k: tvm.tir.trace([y[i][j][k]], "tvm.tir.trace_callback2")
        )
        s = te.create_schedule(z.op)
        f = tvm.build(s, [x, y, z], "llvm")

        xnd = tvm.nd.array(np.ones((n, n, n), dtype=x.dtype))
        ynd = tvm.nd.array(np.zeros((n, n, n), dtype=y.dtype))
        znd = tvm.nd.array(np.zeros((n, n, n), dtype=z.dtype))
        f(xnd, ynd, znd)

        assert np.array_equal(xnd.numpy(), np.ones((n, n, n)))
        assert np.array_equal(ynd.numpy(), np.ones((n, n, n)))
        assert np.array_equal(znd.numpy(), np.ones((n, n, n)))

    for t in ["float64", "float32", "int64", "int32"]:
        check_assign(t)


def test_trace_expr_sum_generated():
    @tvm.register_func("tvm.tir.trace_callback3")
    def trace_buffer(x):
        return

    def check_expr_sum(dtype):
        n = 4
        a = te.placeholder((n, n, n), name="a", dtype=dtype)
        b = te.placeholder((n, n, n), name="b", dtype=dtype)
        c = te.compute(
            a.shape,
            lambda i, j, k: tvm.tir.trace([a[i][j][k]], "tvm.tir.trace_callback3")
            + tvm.tir.trace([b[i][j][k]], "tvm.tir.trace_callback3"),
        )
        s = te.create_schedule(c.op)
        f = tvm.build(s, [a, b, c])
        xnd = tvm.nd.array(np.array(np.ones((n, n, n), dtype=a.dtype)))
        ynd = tvm.nd.array(np.array(np.ones((n, n, n), dtype=b.dtype)))
        znd = tvm.nd.array(np.zeros((n, n, n), dtype=c.dtype))
        f(xnd, ynd, znd)
        assert np.array_equal(znd.numpy(), xnd.numpy() + ynd.numpy())

    for t in ["float64", "float32", "int64", "int32"]:
        check_expr_sum(t)


def test_trace_expr_sum_args():
    @tvm.register_func("tvm.tir.trace_silent")
    def silent(*args):
        return

    def check_expr_sum(dtype):
        n = 4
        a = te.placeholder((n, n, n), name="a", dtype=dtype)
        b = te.placeholder((n, n, n), name="b", dtype=dtype)
        e = te.placeholder((n, n, n), name="e", dtype=dtype)
        d = te.placeholder((n, n, n), name="d", dtype=dtype)

        c = te.compute(
            a.shape,
            lambda i, j, k: tvm.tir.trace([i, j, k, a[i][j][k]], "tvm.tir.trace_silent")
            + tvm.tir.trace([i, j, k, b[i][j][k]], "tvm.tir.trace_silent")
            + tvm.tir.trace([i, j, k, d[i][j][k]], "tvm.tir.trace_silent")
            + tvm.tir.trace([i, j, k, e[i][j][k]], "tvm.tir.trace_silent"),
        )
        s = te.create_schedule(c.op)
        f = tvm.build(s, [a, b, d, e, c])
        a_nd = tvm.nd.array(np.array(np.ones((n, n, n), dtype=a.dtype)))
        b_nd = tvm.nd.array(np.array(np.ones((n, n, n), dtype=b.dtype)))
        d_nd = tvm.nd.array(np.array(np.ones((n, n, n), dtype=d.dtype)))
        e_nd = tvm.nd.array(np.array(np.ones((n, n, n), dtype=e.dtype)))
        c_nd = tvm.nd.array(np.zeros((n, n, n), dtype=c.dtype))
        f(a_nd, b_nd, d_nd, e_nd, c_nd)
        assert np.array_equal(
            c_nd.numpy(), a_nd.numpy() + b_nd.numpy() + d_nd.numpy() + e_nd.numpy()
        )

    for t in ["float64", "float32", "int64", "int32"]:
        check_expr_sum(t)


def test_trace_expr_sum_custom():
    @tvm.register_func("tvm.tir.trace_callback4")
    def trace_buffer(x):
        return

    def check_expr_sum_custom(dtype):
        n = 4
        a = te.placeholder((n, n), name="a", dtype=dtype)
        b = te.placeholder((n, n), name="b", dtype=dtype)
        c = te.compute(
            a.shape,
            lambda i, j: tvm.tir.trace([a[i][j]], "tvm.tir.trace_callback4")
            + tvm.tir.trace([b[i][j]], "tvm.tir.trace_callback4"),
        )
        s = te.create_schedule(c.op)
        f = tvm.build(s, [a, b, c])
        npa = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=a.dtype)
        npb = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=a.dtype)
        xnd = tvm.nd.array(npa)
        ynd = tvm.nd.array(npb)
        znd = tvm.nd.array(np.zeros((n, n), dtype=c.dtype))
        f(xnd, ynd, znd)
        assert np.array_equal(znd.numpy(), npa + npb)

    for t in ["float64", "float32", "int64", "int32"]:
        check_expr_sum_custom(t)


def test_trace_can_change_traced_value_int():
    @tvm.register_func("tvm.tir.trace_change_int_first")
    def trace_buffer(x):
        return 13

    @tvm.register_func("tvm.tir.trace_change_int_second")
    def trace_buffer(x):
        return 14

    def check_assign(dtype):
        n = 4
        x = te.placeholder((n,), name="X", dtype=dtype)
        y = te.compute(x.shape, lambda i: tvm.tir.trace([x[i]], "tvm.tir.trace_change_int_first"))
        z = te.compute(x.shape, lambda i: tvm.tir.trace([y[i]], "tvm.tir.trace_change_int_second"))
        s = te.create_schedule(z.op)
        f = tvm.build(s, [x, y, z], "llvm")

        xnd = tvm.nd.array(np.ones((n,), dtype=x.dtype))
        ynd = tvm.nd.array(np.zeros((n,), dtype=y.dtype))
        znd = tvm.nd.array(np.zeros((n,), dtype=z.dtype))
        f(xnd, ynd, znd)
        check_array_first = np.array([13, 13, 13, 13])
        check_array_second = np.array([14, 14, 14, 14])
        assert np.array_equal(ynd.numpy(), check_array_first)
        assert np.array_equal(znd.numpy(), check_array_second)

    for t in ["int64", "int32"]:
        check_assign(t)


def test_trace_can_change_traced_value_float():
    @tvm.register_func("tvm.tir.trace_change_float_first")
    def trace_buffer(x):
        return 13.0

    @tvm.register_func("tvm.tir.trace_change_float_second")
    def trace_buffer(x):
        return 14.0

    def check_assign(dtype):
        n = 4
        x = te.placeholder((n,), name="X", dtype=dtype)
        y = te.compute(x.shape, lambda i: tvm.tir.trace([x[i]], "tvm.tir.trace_change_float_first"))
        z = te.compute(
            x.shape, lambda i: tvm.tir.trace([y[i]], "tvm.tir.trace_change_float_second")
        )
        s = te.create_schedule(z.op)
        f = tvm.build(s, [x, y, z], "llvm")

        xnd = tvm.nd.array(np.ones((n,), dtype=x.dtype))
        ynd = tvm.nd.array(np.zeros((n,), dtype=y.dtype))
        znd = tvm.nd.array(np.zeros((n,), dtype=z.dtype))
        f(xnd, ynd, znd)
        check_array_first = np.array([13.0, 13.0, 13.0, 13.0])
        check_array_second = np.array([14.0, 14.0, 14.0, 14.0])
        assert np.array_equal(ynd.numpy(), check_array_first)
        assert np.array_equal(znd.numpy(), check_array_second)

    for t in ["float64", "float32"]:
        check_assign(t)


if __name__ == "__main__":
    tvm.testing.main()
