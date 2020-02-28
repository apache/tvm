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

def test_schedule0():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    s = te.create_schedule(A1.op)

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule1():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')

    s = te.create_schedule(A1.op)
    xo, xi = s[A1].split(A1.op.axis[0], 8)
    s[A1].pragma(xo, "auto_unroll_max_step", 10)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule2():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = te.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_scan():
    m = te.var("m")
    n = te.var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    s_state = te.placeholder((m, n))
    s_init = te.compute((1, n), lambda _, i: x[0, i])
    s_update = te.compute((m, n), lambda t, i: s_state[t-1, i] + x[t, i])
    res = tvm.te.scan(s_init, s_update, s_state)

    assert tuple(res.shape) == (m, n)
    s = te.create_schedule(res.op)
    s = s.normalize()
    ir = tvm.lower(s, [s_state], simple_mode=True)
    assert not hasattr(ir.body.body.body.body[1].body.body[1].body, "condition")
    bounds = tvm.te.schedule.InferBound(s)
    assert(bounds[res.op.scan_axis].min.value == 1)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_inline_multi_reduce():
    def argmax_comp(x, y):
        idx = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
        val = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
        return idx, val
    def argmax_init(idx_typ, val_typ):
        return tvm.tir.const(-1, idx_typ), tvm.te.min_value(val_typ)

    argmax = te.comm_reducer(argmax_comp, argmax_init, name='argmax')
    m = te.var('m')
    n = te.var('n')
    val = te.placeholder((m, n), name='val', dtype='float32')
    val1 = te.compute((m, n), lambda i, j: val[i, j]+1, name='val1')
    val2 = te.compute((m, n), lambda i, j: te.exp(val1[i, j]), name='val2')
    k = te.reduce_axis((0, n), 'k')
    T_idx, T_val = te.compute((m, ), lambda i: argmax((k.var, val2[i, k]), axis=k), name='T')
    s = te.create_schedule(T_idx.op)
    s[val1].compute_inline()
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_auto_inline():
    m = te.var('m')
    n = te.var('n')
    A = te.placeholder((m, n), name='A')
    B = te.placeholder((m, n), name='B')
    C = te.placeholder((m, n), name='C')
    T1 = te.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='T1')
    T2 = te.compute((m, n), lambda i, j: T1(i, j) + C(i, j), name='T2')

    s = te.create_schedule(T2.op)
    tvm.te.schedule.AutoInlineElemWise(s)
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_const_bound():
    n = 128
    A = te.placeholder((n,), name='A')
    A1 = te.compute((n,), lambda i: A[i] + 1, name='A1')
    s = te.create_schedule(A1.op)
    xo, xi = s[A1].split(A1.op.axis[0], 8)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_inline_mixed():
    n = te.var('n')
    A = te.placeholder((n, ), name='A')
    A1 = te.compute(A.shape, lambda *i: A(*i) + 1, name='A1')
    A2 = te.compute(A.shape, lambda *i: A1(*i) + 2, name='A2')
    C = te.compute((n,), lambda i: A2[i] + A1[i], name='C')

    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=8)
    s[A1].compute_at(s[C], xo)
    s[A2].compute_inline()
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)
    def check(x):
        if isinstance(x, tvm.tir.Call):
            assert x.func != A2
    tvm.tir.ir_pass.PostOrderVisit(s[C].op.body[0], check)


def test_scan_inline1():
    m = te.var("m")
    n = te.var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    s_state1 = te.placeholder((m, n))
    s_state2 = te.placeholder((m, n))
    s_init1 = te.compute((1, n), lambda _, i: x[0, i])
    s_init2 = te.compute((1, n), lambda _, i: x[0, i])
    s_x1 = te.compute((m, n), lambda t, i: s_state1[t-1, i] + x[t, i], name="x1")
    s_x2 = te.compute((m, n), lambda t, i: s_state2[t-1, i] + 1 , name="x2")
    s_update1 = te.compute((m, n), lambda t, i: s_x1[t, i], "u1")
    s_update2 = te.compute((m, n), lambda t, i: s_x2[t, i], "u2")
    res1, res2 = tvm.te.scan([s_init1, s_init2],
                          [s_update1, s_update2],
                          [s_state1, s_state2])
    s = te.create_schedule(res1.op)
    s[s_x1].compute_inline()
    stmt = tvm.lower(s, [x, res1, res2])


def test_scan_inline2():
    m = te.var("m")
    n = te.var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    s_state1 = te.placeholder((m, n))
    s_state2 = te.placeholder((m, n))
    s_init1 = te.compute((1, n), lambda _, i: x[0, i])
    s_init2 = te.compute((1, n), lambda _, i: x[0, i])
    s_xx = te.compute((m, n), lambda t, i: s_state1[t-1, i] + x[t, i], name="xx")
    s_x1 = te.compute((m, n), lambda t, i: s_xx[t, i] + 1, name="x1")
    s_x2 = te.compute((m, n), lambda t, i: s_xx[t, i] + s_state2[t-1, 2], name="x2")
    s_update1 = te.compute((m, n), lambda t, i: s_x1[t, i], "u1")
    s_update2 = te.compute((m, n), lambda t, i: s_x2[t, i], "u2")
    res1, res2 = tvm.te.scan([s_init1, s_init2],
                          [s_update1, s_update2],
                          [s_state1, s_state2])
    s = te.create_schedule(res1.op)
    s[s_xx].compute_inline()
    s[s_x1].compute_inline()
    s[s_x2].compute_inline()
    stmt = tvm.lower(s, [x, res1, res2])


def test_schedule_cache():
    m = te.var('m')
    n = te.var('n')
    A = te.placeholder((m, n), name='A')
    B = te.placeholder((m, n), name='B')
    C = te.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='C')

    s = te.create_schedule(C.op)
    AA = s.cache_read(A, "shared", readers=[C])
    CC = s.cache_write(C, "shared")
    s[AA].compute_at(s[CC], CC.op.axis[0])
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_middle_cache():
    m = te.var('m')
    n = te.var('n')
    A = te.placeholder((m, n), name='A')
    B = te.placeholder((m, n), name='B')

    C = te.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='C')
    D = te.compute((m, n), lambda i, j:  C(i , j) , name='D')

    s = te.create_schedule(D.op)
    AA = s.cache_read(A, "local", readers=[C])
    BB = s.cache_read(B, "local", readers=[C])
    CC = s.cache_read(C, "local", readers=[D])
    DD = s.cache_write(D, "local")
    #s[AA].compute_at(s[CC], CC.op.axis[0])
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_cache_relayout1():
    m = te.var('m')
    n = te.var('n')
    A = te.placeholder((m, n), name='A')
    B = te.placeholder((m, n), name='B')
    C = te.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='C')

    s = te.create_schedule(C.op)
    s[C].reorder(C.op.axis[1], C.op.axis[0])
    CC = s.cache_write(C, "global")
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_cache_relayout2():
    m = te.var('m')
    n = te.var('n')
    A = te.placeholder((m*4, n), name='A')
    B = te.placeholder((m*4, n), name='B')
    C = te.compute(A.shape, lambda i, j:  A(i, j) * B(i, j), name='C')
    s = te.create_schedule(C.op)
    x, y = C.op.axis
    xo, xi = s[C].split(x, factor=4)
    s[C].reorder(xo, y, xi)
    CC = s.cache_write(C, "global")
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_cache_relayout3():
    m = te.var('m')
    n = te.var('n')
    A = te.placeholder((m*4, n), name='A')
    B = te.placeholder((m*4, n), name='B')
    k = te.reduce_axis((0, n), "k")
    C = te.compute((A.shape[0],),
                    lambda i: te.sum(A(i, k) * B(i, k), axis=k), name='C')
    s = te.create_schedule(C.op)
    x = C.op.axis[0]
    xo, xi = s[C].split(x, factor=4)
    CC = s.cache_write(C, "global")
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_cache_relayout4():
    def _compute(*indice):
        return A(*indice) + 1, B(*indice) / 2
    m = te.var('m')
    n = te.var('n')
    A = te.placeholder((m*4, n), name='A')
    B = te.placeholder((m*4, n), name='B')
    C1, C2 = te.compute(A.shape, _compute, name='C')
    s = te.create_schedule([C1.op, C2.op])
    C1_cache, C2_cache = s.cache_write([C1, C2], "local")
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def intrin_gemv(m, n):
    w = te.placeholder((m, n), name='w')
    x = te.placeholder((n,), name='x')
    k = te.reduce_axis((0, n), name='k')
    z = te.compute((m,), lambda i:
                    te.sum(w[i, k] * x[k], axis=k), name='z')
    Wb = tvm.tir.decl_buffer(w.shape, w.dtype,
                         name="W",
                         offset_factor=16,
                         strides=[te.var('ldw'), 1])
    def intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        ww_ptr = ww.access_ptr("r")
        xx_ptr = xx.access_ptr("r")
        zz_ptr = zz.access_ptr("w")
        body = tvm.tir.call_packed(
            "gemm", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        reset = tvm.tir.call_packed(
            "fill_zero", zz_ptr, n)
        update = tvm.tir.call_packed(
            "gemv_add", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        return body, reset, update

    with tvm.target.build_config(data_alignment=16,
                          offset_factor=16):
        return te.decl_tensor_intrin(z.op, intrin_func,
                                      binds={w: Wb})


def test_schedule_tensor_compute1():
    # basic: split, reorder, tile
    M, N, L = 2048, 1024, 512
    factor, rfactor = 16, 16
    A = te.placeholder((N//factor, L//rfactor, factor, rfactor), name='A')
    B = te.placeholder((M, L//rfactor, rfactor), name='B')
    k = te.reduce_axis((0, L//rfactor), name='k')

    gemv = intrin_gemv(factor, rfactor)
    C = te.compute((N, M//factor, factor),
        lambda i, j: gemv(A[i, k, 0:factor, 0:factor], B[j, k, 0:rfactor], reduce_axis=k),
        name='C')

    s = te.create_schedule(C.op)
    ai, aj, ax = s[C].op.axis
    aio, aii = s[C].split(ai, 16)
    s[C].reorder(aio, aj, aii)
    aioo, ajo, aioi, aji = s[C].tile(aio, aj, 16, 4)

    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def intrin_vadd(n, cache_read=False, cache_write=False):
    scope_ubuf = 'local'
    dtype = 'float32'
    x = te.placeholder((n,), dtype=dtype, name='vx')
    y = te.placeholder((n,), dtype=dtype, name='vy')
    z = te.compute(x.shape, lambda i: x[i] + y[i], name='z')
    s = te.create_schedule(z.op)

    def create_buffer(t):
        return tvm.tir.decl_buffer(t.shape, t.dtype,
                               name='W'+t.name,
                               scope=scope_ubuf,
                               offset_factor=16)

    binds = {}
    if cache_read:
        binds[x] = create_buffer(x)
        binds[y] = create_buffer(y)
    if cache_write:
        binds[z] = create_buffer(z)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        ib.emit(tvm.tir.call_extern(outs[0].dtype, 'vadd', ins[0].access_ptr("r"), ins[1].access_ptr('r'), outs[0].access_ptr('wr')))
        return ib.get()

    with tvm.target.build_config(offset_factor=16):
        return te.decl_tensor_intrin(z.op, intrin_func, binds=binds)


def test_schedule_tensor_compute2():
    # cache_read, cache_write
    M = 1024
    factor = 16
    dtype = 'float32'
    scope_ubuf = 'local'

    A = te.placeholder((M//factor, factor), name="A", dtype=dtype)
    B = te.placeholder((M//factor, factor), name="B", dtype=dtype)

    vadd = intrin_vadd(factor, True, True)
    C = te.compute((M//factor, factor),
        lambda i: vadd(A[i, 0:factor], B[i, 0:factor]), name='C')

    s = te.create_schedule(C.op)
    AL = s.cache_read(A, scope_ubuf, C)
    BL = s.cache_read(B, scope_ubuf, C)
    CL = s.cache_write(C, scope_ubuf)
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_schedule_tensor_compute3():
    # compute_at
    M = 1024
    factor = 16
    dtype = 'float32'
    A = te.placeholder((M//factor, factor), name="A", dtype=dtype)
    B = te.placeholder((M//factor, factor), name="B", dtype=dtype)
    Bi = te.compute((M//factor, factor), lambda i, j: B[i, j] + 5, name="Bi")

    vadd = intrin_vadd(factor)
    C = te.compute((M//factor, factor),
        lambda i: vadd(A[i, 0:factor], Bi[i, 0:factor]), name='C')
    s = te.create_schedule(C.op)
    s[Bi].compute_at(s[C], C.op.axis[0])
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)


def test_loop_dep_reduce():
    X = te.placeholder(shape=(10,), name="x")
    def f(n):
        rv = te.reduce_axis((0, n))
        return te.sum(X[rv], axis=rv)
    Y = te.compute(X.shape, f, name="y")
    s = te.create_schedule([Y.op])
    f = tvm.build(s, [X, Y])


def test_loop_dep_reduce_cache_write():
    X = te.placeholder(shape=(10,), name="x")
    def f(n):
        rv = te.reduce_axis((0, n))
        init = lambda dtype: tvm.tir.Select(n > 1, tvm.tir.const(0, dtype), n.astype(dtype))
        sum = te.comm_reducer(lambda x, y: tvm.te.max(x + y, n.astype('float32')), init, name='sum')
        return sum(X[rv], axis=rv)
    Y = te.compute(X.shape, f, name="y")
    s = te.create_schedule([Y.op])
    s.cache_write(Y, 'local')
    f = tvm.build(s, [X, Y])

def test_reduction_and_dummy_fuse_split():
    n = 10
    X = te.placeholder(shape=(n,), dtype='int32', name="X")
    k = te.reduce_axis((0, n))
    Y = te.compute((), lambda: te.sum(X[k], k), name="Y")
    s = te.create_schedule([Y.op])
    ax = s[Y.op].fuse(*Y.op.axis)
    axo, axi = s[Y.op].split(ax, nparts=20)
    f = tvm.build(s, [Y, X])

    args = [tvm.nd.empty((), 'int32')] + [tvm.nd.array(np.ones((n,), dtype='int32'))]
    f(*args)
    assert args[0].asnumpy() == n

    n = 10
    X = te.placeholder(shape=(n,), dtype='int32', name="X")
    k = te.reduce_axis((0, n))
    Y = te.compute((n,), lambda i: te.sum(X[k], k), name="Y")
    s = te.create_schedule([Y.op])
    ax = s[Y.op].fuse(*(list(Y.op.axis) + list(Y.op.reduce_axis)))
    f = tvm.build(s, [Y, X])

    args = [tvm.nd.array(np.ones((n,), dtype='int32'))] + \
        [tvm.nd.array(np.ones((n,), dtype='int32'))]
    f(*args)
    assert np.all(args[0].asnumpy() == n)

def test_schedule_compute_inline():
    shape = [10, 1024]
    A = te.placeholder(shape, name="A")
    B = te.placeholder(shape, name="B")
    C = te.compute(shape, lambda *index:A(*index)+ B(*index), name = "C")
    def _compute(*index) :
        return C(*index) , C(*index) * B(*index)
    F,E = te.compute(shape, _compute, name = "F")

    s = te.create_schedule([F.op, E.op])
    AL = s.cache_read(A, "local", [C])
    BL = s.cache_read(B, "local", [C,E])
    CL = s.cache_write(C, "local")
    FL, EL = s.cache_write([F, E], "local")
    s[C].compute_inline()

    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

if __name__ == "__main__":
    test_loop_dep_reduce()
    test_loop_dep_reduce_cache_write()
    test_schedule_middle_cache()
    test_inline_multi_reduce()
    test_schedule_cache_relayout4()
    test_schedule_cache_relayout3()
    test_schedule_cache_relayout2()
    test_schedule_cache_relayout1()
    test_schedule_const_bound()
    test_scan_inline1()
    test_scan_inline2()
    test_inline_mixed()
    test_auto_inline()
    test_schedule_scan()
    test_schedule0()
    test_schedule1()
    test_schedule2()
    test_schedule_cache()
    test_schedule_tensor_compute1()
    test_schedule_tensor_compute2()
    test_schedule_tensor_compute3()
    test_reduction_and_dummy_fuse_split()
    test_schedule_compute_inline()
