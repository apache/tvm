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
from nose.tools import raises
import tvm
import numpy as np
def collect_visit(stmt, f):
    ret = []
    tvm.ir_pass.PostOrderVisit(stmt, lambda x: ret.append(f(x)))
    return ret

def lower(sch, args):
    binds = {}
    arg_list = []
    for x in args:
        if isinstance(x, tvm.tensor.Tensor):
            buf = tvm.decl_buffer(x.shape, dtype=x.dtype, name=x.name)
            assert x not in binds
            binds[x] = buf
            arg_list.append(buf)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    sch = sch.normalize()
    bounds = tvm.schedule.InferBound(sch)
    stmt = tvm.schedule.ScheduleOps(sch, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
    stmt = tvm.ir_pass.StorageFlatten(stmt, binds, 64, True)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.VectorizeLoop(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    return stmt

@raises(Exception)
def test_out_of_bounds_llvm(index_a, index_b):
    n = tvm.var("n")
    A = tvm.placeholder ((n,), name='A')
    B = tvm.placeholder ((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i + index_a] + B[i + index_b], name='C')
    s = tvm.create_schedule (C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower (s, [A, B, C], simple_mode=True)
    print (stmt)
    fadd = tvm.build (s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
    ctx = tvm.context(tgt, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), ctx)
    fadd (a, b, c)

def test_in_bounds_llvm():
    n = tvm.var("n")
    A = tvm.placeholder ((n,), name='A')
    B = tvm.placeholder ((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule (C.op)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower (s, [A, B, C], simple_mode=True)
    print (stmt)
    fadd = tvm.build (s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
    ctx = tvm.context(tgt, 0)
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=1024).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(1024, dtype=C.dtype), ctx)
    fadd (a, b, c)

@raises(Exception)
def test_out_of_bounds_vectorize_llvm(nn, index_a, index_b):
    n = tvm.convert(nn)
    a = tvm.placeholder((n), name='a')
    b = tvm.placeholder((n), name='b')
    c = tvm.compute((n,), lambda i: a[i + index_a] + b[i + index_b], name='c')
    s = tvm.create_schedule(c.op)
    xo, xi = s[c].split(c.op.axis[0], factor=8)
    s[c].parallel(xo)
    s[c].vectorize(xi)
    tgt = "llvm"
    tgt_host = "llvm"
    stmt = tvm.lower (s, [a, b, c], simple_mode=True)
    print (stmt)
    f = tvm.build(s, [a, b, c], tgt, target_host=tgt_host, name="myaddvec")
    ctx = tvm.cpu(0)
    n = nn
    a = tvm.nd.array(np.random.uniform(size=(n)).astype(a.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(n)).astype(a.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=c.dtype), ctx)
    f(a, b, c)

def test_in_bounds_vectorize_llvm():
    n = 512
    lanes = 2
    A = tvm.placeholder((n,), name='A', dtype="float32x%d" % lanes)
    B = tvm.compute((n,), lambda i: A[i], name='B')
    C = tvm.compute((n,), lambda i: B[i] + tvm.const(1, A.dtype), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], nparts=2)
    _, xi = s[C].split(xi, factor=2)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    s[B].compute_at(s[C], xo)
    xo, xi = s[B].split(B.op.axis[0], factor=2)
    s[B].vectorize(xi)
    # build and invoke the kernel.
    lowered_func = tvm.lower (s, [A, C], "llvm", simple_mode=False)
    print (lowered_func.body)
    f = tvm.build(s, [A, C], "llvm")
    ctx = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.empty((n,), A.dtype).copyfrom(
        np.random.uniform(size=(n, lanes)))
    c = tvm.nd.empty((n,), C.dtype, ctx)
    f(a, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

def test_in_bounds_loop_partition_basic_llvm():
    n = tvm.var('n')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i]+B[i])
    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32,)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(32,)).astype(B.dtype), ctx)
    t = tvm.nd.empty((32,), T.dtype, ctx)
    f(a, b, t)

@raises(Exception)
def test_out_of_bounds_loop_partition_basic_llvm(index_a, index_b):
    n = tvm.var('n')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i + index_a]+B[i + index_b])
    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32,)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(32,)).astype(B.dtype), ctx)
    t = tvm.nd.empty((32,), T.dtype, ctx)
    f(a, b, t)

def test_in_bounds_const_loop_partition_ir():
    def check_attr_stmt (x):
        if isinstance(x, tvm.stmt.AttrStmt) and x.attr_key == "buffer_bound" and str(x.value) == str(n):
            return True
        return False

    def check_branch_stmt (x):
        if isinstance(x, tvm.stmt.IfThenElse):
            return True
        return False

    def assert_bound_instrumentation(stmt, f, nums):
        count = 0
        for i in collect_visit(stmt, f):
            if i is True:
              count = count + 1
        assert (count == nums)

    def collect_branch_stmt (x):
        if isinstance(x, tvm.stmt.IfThenElse):
            branch_collector.append(x)

    n = 21
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i]+B[i])
    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.schedule.InferBound(s)
    stmt = lower (s, [A, B, T])
    # num_attributes = num_buffers * num_splits = 2 * 3
    # before instrumentation
    assert_bound_instrumentation(stmt, check_attr_stmt, 2 * 3)
    assert_bound_instrumentation(stmt, check_branch_stmt, 0)
    stmt = tvm.ir_pass.InstrumentBoundCheckers(stmt)
    # after instrumentation
    assert_bound_instrumentation(stmt, check_attr_stmt, 2 * 3)
    assert_bound_instrumentation(stmt, check_branch_stmt, 2)
    print (stmt)
    branch_collector = list()
    collect_visit(stmt, collect_branch_stmt)
    assert(len(branch_collector) ==  2)
    print (branch_collector[0].condition)
    print (branch_collector[1].condition)

def test_in_bounds_const_loop_partition_llvm():
    with tvm.build_config(instrument_bound_checkers=True, partition_const_loop=True):
        n = 21
        A = tvm.placeholder((n, ), name='A')
        B = tvm.placeholder((n, ), name='B')

        T = tvm.compute((n, ), lambda i: A[i]+B[i])
        s = tvm.create_schedule(T.op)
        xo, xi = s[T].split(T.op.axis[0], factor=4)
        lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
        print (lowered_func.body)
        ctx = tvm.cpu(0)

        f = tvm.build(s, [A, B, T], "llvm")
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), ctx)
        t = tvm.nd.empty((n,), T.dtype, ctx)
        f(a, b, t)

@raises(Exception)
def test_out_of_bounds_const_loop_partition_llvm(index_a, index_b):
    with tvm.build_config(instrument_bound_checkers=True, partition_const_loop=True):
        n = 21
        A = tvm.placeholder((n, ), name='A')
        B = tvm.placeholder((n, ), name='B')

        T = tvm.compute((n, ), lambda i: A[i + index_a]+B[i + index_b])
        s = tvm.create_schedule(T.op)
        xo, xi = s[T].split(T.op.axis[0], factor=4)
        lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
        print (lowered_func.body)
        ctx = tvm.cpu(0)

        f = tvm.build(s, [A, B, T], "llvm")
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), ctx)
        t = tvm.nd.empty((n,), T.dtype, ctx)
        f(a, b, t)

def test_in_bounds_conv_llvm(loop_tiling=False):
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = tvm.placeholder((batch_size, in_channel, in_height, in_width), name='data')
    kernel = tvm.placeholder((kernel_height, kernel_width, in_channel,
        out_channel), name='kernel')
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')
    conv = tvm.compute((batch_size, out_channel, out_height, out_width),
                       lambda n, oc, oh, ow: tvm.sum(data[n, ic, oh*HSTR + kh, ow*WSTR + kw] *
                                                     kernel[kh, kw, ic, oc],
                                                     axis=[ic, kh, kw]),
                       name="conv2d")
    s = tvm.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    if loop_tiling:
        oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    lowered_func = tvm.lower(s, [data, kernel, conv], simple_mode=True)
    print (lowered_func.body)
    ctx = tvm.cpu (0)

    f = tvm.build(s, [data, kernel, conv], "llvm")
    data_input = tvm.nd.array(np.random.uniform(
          size=(batch_size, in_channel, in_height, in_width)).astype(tvm.float32), ctx)
    kernel_input = tvm.nd.array(np.random.uniform(
          size=(kernel_height, kernel_width, in_channel, out_channel)).astype(tvm.float32), ctx)
    conv_out = tvm.nd.empty ((batch_size, out_channel, out_height, out_width), tvm.float32, ctx)
    f(data_input, kernel_input, conv_out)

@raises(Exception)
def test_out_of_bounds_conv_llvm(data_offsets, kernel_offsets, loop_tiling=False):
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = tvm.placeholder((batch_size, in_channel, in_height, in_width), name='data')
    kernel = tvm.placeholder((kernel_height, kernel_width, in_channel,
        out_channel), name='kernel')
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')
    conv = tvm.compute((batch_size, out_channel, out_height, out_width),
                       lambda n, oc, oh, ow: tvm.sum(data[n + data_offsets[0],
                                                          ic + data_offsets[1],
                                                          oh*HSTR + kh + data_offsets[2],
                                                          ow*WSTR + kw + data_offsets[3]]
                                                          *
                                                     kernel[kh + kernel_offsets[0],
                                                     kw + kernel_offsets[1],
                                                     ic + kernel_offsets[2],
                                                     oc + kernel_offsets[3]],
                                                     axis=[ic, kh, kw]),
                       name="conv2d")
    s = tvm.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    if loop_tiling:
        oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    lowered_func = tvm.lower(s, [data, kernel, conv], simple_mode=True)
    print (lowered_func.body)
    ctx = tvm.cpu (0)

    f = tvm.build(s, [data, kernel, conv], "llvm")
    data_input = tvm.nd.array(np.random.uniform(
          size=(batch_size, in_channel, in_height, in_width)).astype(tvm.float32), ctx)
    kernel_input = tvm.nd.array(np.random.uniform(
          size=(kernel_height, kernel_width, in_channel, out_channel)).astype(tvm.float32), ctx)
    conv_out = tvm.nd.empty ((batch_size, out_channel, out_height, out_width), tvm.float32, ctx)
    f(data_input, kernel_input, conv_out)

def test_in_bounds_tensors_with_same_shapes1D_llvm():
    n = tvm.var('n')
    k = tvm.var('k')
    m = tvm.var('m')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((k, ), name='B')

    T = tvm.compute((m, ), lambda i: A[i]*B[i])
    s = tvm.create_schedule(T.op)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32, )).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(32,)).astype(B.dtype), ctx)
    t = tvm.nd.empty((32,), T.dtype, ctx)
    f(a, b, t)

@raises(Exception)
def test_out_of_bounds_tensors_with_diff_shapes1D_llvm(a_shape, b_shape, c_shape):
    n = tvm.var('n')
    k = tvm.var('k')
    m = tvm.var('m')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((k, ), name='B')

    T = tvm.compute((m, ), lambda i: A[i]*B[i])
    s = tvm.create_schedule(T.op)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(a_shape,)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(b_shape,)).astype(B.dtype), ctx)
    t = tvm.nd.empty((c_shape,), T.dtype, ctx)
    f(a, b, t)

def test_in_bounds_tensors_with_same_shapes2D_llvm():
    n = tvm.var('n')
    k = tvm.var('k')
    m = tvm.var('m')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((k, k), name='B')

    T = tvm.compute((m, m), lambda i, j: A[i][j]*B[i][j])
    s = tvm.create_schedule(T.op)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32, 32)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(32, 32)).astype(B.dtype), ctx)
    t = tvm.nd.empty((32, 32), T.dtype, ctx)
    f(a, b, t)

@raises(Exception)
def test_out_of_bounds_tensors_with_diff_shapes2D_llvm(a_shape, b_shape, c_shape):
    n = tvm.var('n')
    k = tvm.var('k')
    m = tvm.var('m')
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((k, k), name='B')

    T = tvm.compute((m, m), lambda i, j: A[i][j]*B[i][j])
    s = tvm.create_schedule(T.op)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(a_shape[0],a_shape[1])).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(b_shape[0],b_shape[1])).astype(B.dtype), ctx)
    t = tvm.nd.empty((c_shape[0],c_shape[1]), T.dtype, ctx)
    f(a, b, t)

def test_in_bounds_tensors_with_same_shapes3D_llvm():
    n = tvm.var('n')
    k = tvm.var('k')
    m = tvm.var('m')
    A = tvm.placeholder((n, n, n), name='A')
    B = tvm.placeholder((k, k, k), name='B')

    T = tvm.compute((m, m, m), lambda i, j, p: A[i][j][p]*B[i][j][p])
    s = tvm.create_schedule(T.op)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(32,32,32)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(32,32,32)).astype(B.dtype), ctx)
    t = tvm.nd.empty((32, 32, 32), T.dtype, ctx)
    f(a, b, t)

@raises(Exception)
def test_out_of_bounds_tensors_with_diff_shapes3D_llvm(a_shape, b_shape, c_shape):
    n = tvm.var('n')
    k = tvm.var('k')
    m = tvm.var('m')
    A = tvm.placeholder((n, n, n), name='A')
    B = tvm.placeholder((k, k, k), name='B')

    T = tvm.compute((m, m, m), lambda i, j, p: A[i][j][p]*B[i][j][p])
    s = tvm.create_schedule(T.op)
    lowered_func = tvm.lower (s, [A, B, T], "llvm", simple_mode=False)
    print (lowered_func.body)
    ctx = tvm.cpu(0)

    f = tvm.build(s, [A, B, T], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(a_shape[0],a_shape[1], c_shape[2])).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(b_shape[0],b_shape[1], b_shape[2])).astype(B.dtype), ctx)
    t = tvm.nd.empty((c_shape[0],c_shape[1],c_shape[2]), T.dtype, ctx)
    f(a, b, t)

@raises(Exception)
def test_out_of_bounds_tensors_with_zero_shape_op_with_not_zero_shape_llvm():
    if not tvm.module.enabled("llvm"):
        return
    n = 64
    A = tvm.placeholder((n, ), name='A')
    scale = tvm.placeholder((), name='scale')
    k = tvm.reduce_axis((0, n), name="k")
    C = tvm.compute((), lambda : tvm.sum(A[k + k + k] * scale, axis=k), name="C")
    D = tvm.compute((), lambda : C + 1)
    s = tvm.create_schedule(D.op)
    stmt = tvm.lower (s, [A, scale, D], simple_mode=True)
    print (stmt)
    # build and invoke the kernel.
    f = tvm.build(s, [A, scale, D], "llvm")
    ctx = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
    sc = tvm.nd.array(
        np.random.randint(0, 2, size=()).astype(scale.dtype), ctx)
    d = tvm.nd.empty((), D.dtype, ctx)
    f(a, sc, d)
    d_np = np.sum(a.asnumpy()) * sc.asnumpy() + 1
    tvm.testing.assert_allclose(d.asnumpy(), d_np)

if __name__ == "__main__":
    with tvm.build_config(instrument_bound_checkers=True):
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
        test_out_of_bounds_tensors_with_diff_shapes1D_llvm (32, 64, 64)
        test_out_of_bounds_tensors_with_diff_shapes1D_llvm (64, 32, 64)
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
