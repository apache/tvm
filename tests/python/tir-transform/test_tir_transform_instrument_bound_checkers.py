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
from tvm import te, tir

import pytest
import numpy as np


def collect_visit(stmt, f):
    ret = []
    tvm.tir.stmt_functor.post_order_visit(stmt, lambda x: ret.append(f(x)))
    return ret


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_llvm(index_a, index_b):
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i + index_a] + B[i + index_b], name="C")
    s = te.create_schedule(C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower(s, [A, B, C], simple_mode=True)
    print(stmt)
    tgt = tvm.target.Target(tgt, tgt_host)
    fadd = tvm.build(s, [A, B, C], target=tgt, name="myadd")
    dev = tvm.device(tgt.kind.name, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), dev)
    fadd(a, b, c)


@tvm.testing.requires_llvm
def test_in_bounds_llvm():
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower(s, [A, B, C], simple_mode=True)
    tgt = tvm.target.Target(tgt, tgt_host)
    fadd = tvm.build(s, [A, B, C], target=tgt, name="myadd")
    dev = tvm.device(tgt.kind.name, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), dev)
    fadd(a, b, c)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_vectorize_llvm(nn, index_a, index_b):
    n = tvm.runtime.convert(nn)
    a = te.placeholder((n), name="a")
    b = te.placeholder((n), name="b")
    c = te.compute((n,), lambda i: a[i + index_a] + b[i + index_b], name="c")
    s = te.create_schedule(c.op)
    xo, xi = s[c].split(c.op.axis[0], factor=8)
    s[c].parallel(xo)
    s[c].vectorize(xi)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower(s, [a, b, c], simple_mode=True)
    tgt = tvm.target.Target(tgt, tgt_host)
    f = tvm.build(s, [a, b, c], target=tgt, name="myaddvec")
    dev = tvm.cpu(0)
    n = nn
    a = tvm.nd.array(np.random.uniform(size=(n)).astype(a.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(n)).astype(a.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=c.dtype), dev)
    f(a, b, c)


@tvm.testing.requires_llvm
def test_in_bounds_vectorize_llvm():
    n = 512
    lanes = 2
    A = te.placeholder((n,), name="A", dtype="float32x%d" % lanes)
    B = te.compute((n,), lambda i: A[i], name="B")
    C = te.compute((n,), lambda i: B[i] + tvm.tir.const(1, A.dtype), name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], nparts=2)
    _, xi = s[C].split(xi, factor=2)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    s[B].compute_at(s[C], xo)
    xo, xi = s[B].split(B.op.axis[0], factor=2)
    s[B].vectorize(xi)
    # build and invoke the kernel.
    lowered_func = tvm.lower(s, [A, C], "llvm", simple_mode=False)
    f = tvm.build(s, [A, C], "llvm")
    dev = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.empty((n,), A.dtype).copyfrom(
        np.random.uniform(size=[n] + ([] if lanes == 1 else [lanes]))
    )
    c = tvm.nd.empty((n,), C.dtype, dev)
    f(a, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)


@tvm.testing.requires_llvm
def test_in_bounds_loop_partition_basic_llvm():
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    T = te.compute((n,), lambda i: A[i] + B[i])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32,)).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(32,)).astype(B.dtype), dev)
    t = tvm.nd.empty((32,), T.dtype, dev)
    f(a, b, t)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_loop_partition_basic_llvm(index_a, index_b):
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    T = te.compute((n,), lambda i: A[i + index_a] + B[i + index_b])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32,)).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(32,)).astype(B.dtype), dev)
    t = tvm.nd.empty((32,), T.dtype, dev)
    f(a, b, t)


def test_in_bounds_const_loop_partition_ir():
    def check_attr_stmt(x):
        if (
            isinstance(x, tvm.tir.AttrStmt)
            and x.attr_key == "buffer_bound"
            and tvm.ir.structural_equal(x.value.args, [n])
        ):
            return True
        return False

    def check_branch_stmt(x):
        if isinstance(x, tvm.tir.IfThenElse):
            return True
        return False

    def assert_bound_instrumentation(stmt, f, nums):
        count = 0
        for i in collect_visit(stmt, f):
            if i is True:
                count = count + 1
        assert count == nums

    def collect_branch_stmt(x):
        if isinstance(x, tvm.tir.IfThenElse):
            branch_collector.append(x)

    n = tir.const(21)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    T = te.compute((n,), lambda i: A[i] + B[i])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    with tvm.transform.PassContext(
        config={
            "tir.instrument_bound_checkers": True,
            "tir.LoopPartition": {"partition_const_loop": True},
        }
    ):
        mod = tvm.driver.lower(s, [A, B, T], name="main")

    stmt = mod["main"].body
    # after instrumentation
    assert_bound_instrumentation(stmt, check_attr_stmt, 2 * 3)
    assert_bound_instrumentation(stmt, check_branch_stmt, 2)

    branch_collector = list()
    collect_visit(stmt, collect_branch_stmt)
    assert len(branch_collector) == 2


@tvm.testing.requires_llvm
def test_in_bounds_const_loop_partition_llvm():
    with tvm.transform.PassContext(
        config={
            "tir.instrument_bound_checkers": True,
            "tir.LoopPartition": {"partition_const_loop": True},
        }
    ):
        n = 21
        A = te.placeholder((n,), name="A")
        B = te.placeholder((n,), name="B")

        T = te.compute((n,), lambda i: A[i] + B[i])
        s = te.create_schedule(T.op)
        xo, xi = s[T].split(T.op.axis[0], factor=4)
        lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
        dev = tvm.cpu(0)

        f = tvm.build(s, [A, B, T], "llvm")
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), dev)
        t = tvm.nd.empty((n,), T.dtype, dev)
        f(a, b, t)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_const_loop_partition_llvm(index_a, index_b):
    with tvm.transform.PassContext(
        config={
            "tir.instrument_bound_checkers": True,
            "tir.LoopPartition": {"partition_const_loop": True},
        }
    ):
        n = 21
        A = te.placeholder((n,), name="A")
        B = te.placeholder((n,), name="B")

        T = te.compute((n,), lambda i: A[i + index_a] + B[i + index_b])
        s = te.create_schedule(T.op)
        xo, xi = s[T].split(T.op.axis[0], factor=4)
        lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
        dev = tvm.cpu(0)

        f = tvm.build(s, [A, B, T], "llvm")
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), dev)
        t = tvm.nd.empty((n,), T.dtype, dev)
        f(a, b, t)


@tvm.testing.requires_llvm
def test_in_bounds_conv_llvm(loop_tiling=False):
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = te.placeholder((batch_size, in_channel, in_height, in_width), name="data")
    kernel = te.placeholder((kernel_height, kernel_width, in_channel, out_channel), name="kernel")
    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")
    conv = te.compute(
        (batch_size, out_channel, out_height, out_width),
        lambda n, oc, oh, ow: te.sum(
            data[n, ic, oh * HSTR + kh, ow * WSTR + kw] * kernel[kh, kw, ic, oc], axis=[ic, kh, kw]
        ),
        name="conv2d",
    )
    s = te.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    if loop_tiling:
        oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    lowered_func = tvm.lower(s, [data, kernel, conv], simple_mode=True)
    dev = tvm.cpu(0)

    f = tvm.build(s, [data, kernel, conv], "llvm")
    data_input = tvm.nd.array(
        np.random.uniform(size=(batch_size, in_channel, in_height, in_width)).astype("float32"), dev
    )
    kernel_input = tvm.nd.array(
        np.random.uniform(size=(kernel_height, kernel_width, in_channel, out_channel)).astype(
            "float32"
        ),
        dev,
    )
    conv_out = tvm.nd.empty((batch_size, out_channel, out_height, out_width), "float32", dev)
    f(data_input, kernel_input, conv_out)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_conv_llvm(data_offsets, kernel_offsets, loop_tiling=False):
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = te.placeholder((batch_size, in_channel, in_height, in_width), name="data")
    kernel = te.placeholder((kernel_height, kernel_width, in_channel, out_channel), name="kernel")
    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")
    conv = te.compute(
        (batch_size, out_channel, out_height, out_width),
        lambda n, oc, oh, ow: te.sum(
            data[
                n + data_offsets[0],
                ic + data_offsets[1],
                oh * HSTR + kh + data_offsets[2],
                ow * WSTR + kw + data_offsets[3],
            ]
            * kernel[
                kh + kernel_offsets[0],
                kw + kernel_offsets[1],
                ic + kernel_offsets[2],
                oc + kernel_offsets[3],
            ],
            axis=[ic, kh, kw],
        ),
        name="conv2d",
    )
    s = te.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    if loop_tiling:
        oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    lowered_func = tvm.lower(s, [data, kernel, conv], simple_mode=True)
    dev = tvm.cpu(0)

    f = tvm.build(s, [data, kernel, conv], "llvm")
    data_input = tvm.nd.array(
        np.random.uniform(size=(batch_size, in_channel, in_height, in_width)).astype("float32"), dev
    )
    kernel_input = tvm.nd.array(
        np.random.uniform(size=(kernel_height, kernel_width, in_channel, out_channel)).astype(
            "float32"
        ),
        dev,
    )
    conv_out = tvm.nd.empty((batch_size, out_channel, out_height, out_width), "float32", dev)
    f(data_input, kernel_input, conv_out)


@tvm.testing.requires_llvm
def test_in_bounds_tensors_with_same_shapes1D_llvm():
    n = te.size_var("n")
    k = te.size_var("k")
    m = te.size_var("m")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((k,), name="B")

    T = te.compute((m,), lambda i: A[i] * B[i])
    s = te.create_schedule(T.op)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32,)).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(32,)).astype(B.dtype), dev)
    t = tvm.nd.empty((32,), T.dtype, dev)
    f(a, b, t)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_tensors_with_diff_shapes1D_llvm(a_shape, b_shape, c_shape):
    n = te.size_var("n")
    k = te.size_var("k")
    m = te.size_var("m")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((k,), name="B")

    T = te.compute((m,), lambda i: A[i] * B[i])
    s = te.create_schedule(T.op)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(a_shape,)).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(b_shape,)).astype(B.dtype), dev)
    t = tvm.nd.empty((c_shape,), T.dtype, dev)
    f(a, b, t)


@tvm.testing.requires_llvm
def test_in_bounds_tensors_with_same_shapes2D_llvm():
    n = te.size_var("n")
    k = te.size_var("k")
    m = te.size_var("m")
    A = te.placeholder((n, n), name="A")
    B = te.placeholder((k, k), name="B")

    T = te.compute((m, m), lambda i, j: A[i][j] * B[i][j])
    s = te.create_schedule(T.op)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32, 32)).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(32, 32)).astype(B.dtype), dev)
    t = tvm.nd.empty((32, 32), T.dtype, dev)
    f(a, b, t)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_tensors_with_diff_shapes2D_llvm(a_shape, b_shape, c_shape):
    n = te.size_var("n")
    k = te.size_var("k")
    m = te.size_var("m")
    A = te.placeholder((n, n), name="A")
    B = te.placeholder((k, k), name="B")

    T = te.compute((m, m), lambda i, j: A[i][j] * B[i][j])
    s = te.create_schedule(T.op)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(a_shape[0], a_shape[1])).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(b_shape[0], b_shape[1])).astype(B.dtype), dev)
    t = tvm.nd.empty((c_shape[0], c_shape[1]), T.dtype, dev)
    f(a, b, t)


@tvm.testing.requires_llvm
def test_in_bounds_tensors_with_same_shapes3D_llvm():
    n = te.size_var("n")
    k = te.size_var("k")
    m = te.size_var("m")
    A = te.placeholder((n, n, n), name="A")
    B = te.placeholder((k, k, k), name="B")

    T = te.compute((m, m, m), lambda i, j, p: A[i][j][p] * B[i][j][p])
    s = te.create_schedule(T.op)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)

    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32, 32, 32)).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=(32, 32, 32)).astype(B.dtype), dev)
    t = tvm.nd.empty((32, 32, 32), T.dtype, dev)
    f(a, b, t)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_tensors_with_diff_shapes3D_llvm(a_shape, b_shape, c_shape):
    n = te.size_var("n")
    k = te.size_var("k")
    m = te.size_var("m")
    A = te.placeholder((n, n, n), name="A")
    B = te.placeholder((k, k, k), name="B")

    T = te.compute((m, m, m), lambda i, j, p: A[i][j][p] * B[i][j][p])
    s = te.create_schedule(T.op)
    lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)

    dev = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(
        np.random.uniform(size=(a_shape[0], a_shape[1], c_shape[2])).astype(A.dtype), dev
    )
    b = tvm.nd.array(
        np.random.uniform(size=(b_shape[0], b_shape[1], b_shape[2])).astype(B.dtype), dev
    )
    t = tvm.nd.empty((c_shape[0], c_shape[1], c_shape[2]), T.dtype, dev)
    f(a, b, t)


@tvm.testing.requires_llvm
@pytest.mark.xfail
def test_out_of_bounds_tensors_with_zero_shape_op_with_not_zero_shape_llvm():
    n = 64
    A = te.placeholder((n,), name="A")
    scale = te.placeholder((), name="scale")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((), lambda: te.sum(A[k + k + k] * scale, axis=k), name="C")
    D = te.compute((), lambda: C + 1)
    s = te.create_schedule(D.op)
    stmt = tvm.lower(s, [A, scale, D], simple_mode=True)

    # build and invoke the kernel.
    f = tvm.build(s, [A, scale, D], "llvm")
    dev = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), dev)
    sc = tvm.nd.array(np.random.randint(0, 2, size=()).astype(scale.dtype), dev)
    d = tvm.nd.empty((), D.dtype, dev)
    f(a, sc, d)
    d_np = np.sum(a.numpy()) * sc.numpy() + 1
    tvm.testing.assert_allclose(d.numpy(), d_np)


if __name__ == "__main__":
    with tvm.transform.PassContext(
        config={
            "tir.instrument_bound_checkers": True,
        }
    ):
        # zero scale
        test_out_of_bounds_tensors_with_zero_shape_op_with_not_zero_shape_llvm()
        # in bound
        test_in_bounds_llvm()
        # upper bound
        test_out_of_bounds_llvm(1, 0)
        test_out_of_bounds_llvm(0, 1)
        test_out_of_bounds_llvm(1, 1)
        test_out_of_bounds_llvm(10000, 0)
        test_out_of_bounds_llvm(0, 10000)
        test_out_of_bounds_llvm(10000, 10000)
        # lower bound
        test_out_of_bounds_llvm(-1, 0)
        test_out_of_bounds_llvm(0, -1)
        test_out_of_bounds_llvm(-1, -1)
        test_out_of_bounds_llvm(-10000, 0)
        test_out_of_bounds_llvm(0, -10000)
        test_out_of_bounds_llvm(-10000, -10000)
        # vectorize in bound
        test_in_bounds_vectorize_llvm()
        # vectorization upper bound
        test_out_of_bounds_vectorize_llvm(1024, 1000, 0)
        test_out_of_bounds_vectorize_llvm(1024, 0, 10000)
        # vectorization lower bound
        test_out_of_bounds_vectorize_llvm(1024, -1000, 0)
        test_out_of_bounds_vectorize_llvm(1024, 0, -10000)
        test_in_bounds_const_loop_partition_llvm()
        test_out_of_bounds_const_loop_partition_llvm(1, 0)
        test_out_of_bounds_const_loop_partition_llvm(0, 1)
        test_out_of_bounds_const_loop_partition_llvm(-1, 0)
        test_out_of_bounds_const_loop_partition_llvm(0, -1)
        test_in_bounds_loop_partition_basic_llvm()
        test_out_of_bounds_loop_partition_basic_llvm(32, 0)
        test_out_of_bounds_loop_partition_basic_llvm(0, 32)
        test_out_of_bounds_loop_partition_basic_llvm(-32, 0)
        test_out_of_bounds_loop_partition_basic_llvm(0, -32)
        # conv
        test_in_bounds_conv_llvm()
        test_out_of_bounds_conv_llvm([1, 0, 0, 0], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 1, 0, 0], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 1, 0], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 1], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([-1, 0, 0, 0], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, -1, 0, 0], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, -1, 0], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, -1], [0, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [1, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 1, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, 1, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, 0, 1])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [-1, 0, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, -1, 0, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, -1, 0])
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, 0, -1])
        # loop tiling
        test_in_bounds_conv_llvm(True)
        test_out_of_bounds_conv_llvm([1, 0, 0, 0], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 1, 0, 0], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 1, 0], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 1], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([-1, 0, 0, 0], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, -1, 0, 0], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, -1, 0], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, -1], [0, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [1, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 1, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, 1, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, 0, 1], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [-1, 0, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, -1, 0, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, -1, 0], True)
        test_out_of_bounds_conv_llvm([0, 0, 0, 0], [0, 0, 0, -1], True)
        # tensors with diff shapes basic operation such as mul
        test_out_of_bounds_tensors_with_diff_shapes1D_llvm(32, 64, 64)
        test_out_of_bounds_tensors_with_diff_shapes1D_llvm(64, 32, 64)
        test_out_of_bounds_tensors_with_diff_shapes2D_llvm([64, 64], [32, 32], [64, 64])
        test_out_of_bounds_tensors_with_diff_shapes2D_llvm([32, 32], [64, 64], [64, 64])
        test_out_of_bounds_tensors_with_diff_shapes3D_llvm([64, 64, 64], [32, 32, 32], [64, 64, 64])
        test_out_of_bounds_tensors_with_diff_shapes3D_llvm([32, 32, 32], [64, 64, 64], [64, 64, 64])
        # check tensors with the same shapes
        test_in_bounds_tensors_with_same_shapes1D_llvm()
        test_in_bounds_tensors_with_same_shapes2D_llvm()
        test_in_bounds_tensors_with_same_shapes3D_llvm()
        # ir tests
        test_in_bounds_const_loop_partition_ir()
