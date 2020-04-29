
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
import topi
import unittest
from tvm.contrib.nvcc import have_fp16, have_int8
from tvm.contrib import nvcc

tx = te.thread_axis("threadIdx.x")
bx = te.thread_axis("blockIdx.x")

def test_cuda_vectorize_add():
    num_thread = 8
    def check_cuda(dtype, n, lanes):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        if dtype == "int8" and not have_int8(tvm.gpu(0).compute_version):
            print("skip because gpu does not support int8")
            return
        A = te.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name='B')
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, bx)
        s[B].bind(xi, tx)
        fun = tvm.build(s, [A, B], "cuda")
        ctx = tvm.gpu(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(
            np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

    check_cuda("float32", 64, 2)
    check_cuda("float32", 64, 3)
    check_cuda("float32", 64, 4)
    check_cuda("int8",    64, 2)
    check_cuda("int8",    64, 3)
    check_cuda("int8",    64, 4)
    check_cuda("uint8",   64, 2)
    check_cuda("uint8",   64, 3)
    check_cuda("uint8",   64, 4)
    check_cuda("float16", 64, 2)
    check_cuda("float16", 64, 4)
    check_cuda("float16", 64, 6)
    check_cuda("float16", 64, 8)

def test_cuda_multiply_add():
    num_thread = 8
    def check_cuda(dtype, n, lanes):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "int8" and not have_int8(tvm.gpu(0).compute_version):
            print("skip because gpu does not support int8")
            return
        A = te.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = te.placeholder((n,), name='B', dtype="%sx%d" % (dtype, lanes))
        C = te.placeholder((n,), name='C', dtype="int32")
        D = te.compute((n,),
                        lambda i: tvm.tir.call_pure_extern("int32", "__dp4a", A[i], B[i], C[i]), name='D')
        s = te.create_schedule(D.op)
        xo, xi = s[D].split(D.op.axis[0], factor=num_thread)
        s[D].bind(xo, bx)
        s[D].bind(xi, tx)
        fun = tvm.build(s, [A, B, C, D], "cuda")
        np_a = np.random.randint(low=-128, high=127, size=(n,lanes))
        np_b = np.random.randint(low=-128, high=127, size=(n,lanes))
        np_c = np.random.randint(low=0, high=127, size=(n,))
        np_d = [sum(x * y) + z for x, y, z in zip(np_a, np_b, np_c)]
        ctx = tvm.gpu(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, ctx).copyfrom(np_b)
        c = tvm.nd.empty((n,), C.dtype, ctx).copyfrom(np_c)
        d = tvm.nd.empty((n,), D.dtype, ctx)
        fun(a, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), np_d)
    check_cuda("int8", 64, 4)

def test_cuda_vectorize_load():
    num_thread = 8
    def check_cuda(dtype, n, lanes):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        ctx = tvm.gpu(0)
        A = te.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i], name='B')
        s = te.create_schedule(B.op)
        block, thread = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(block, bx)
        s[B].bind(thread, tx)
        fun = tvm.build(s, [A, B], "cuda", name="vector_load")
        np_a = np.random.randint(low=-128, high=127, size=(n,lanes))
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a,b)
        tvm.testing.assert_allclose(a.asnumpy(), b.asnumpy())
    check_cuda("int8", 64, 2)
    check_cuda("int8", 64, 3)
    check_cuda("int8", 64, 4)
    check_cuda("int8", 64, 8)
    check_cuda("int8", 64, 16)

def test_cuda_make_int8():
    def check_cuda(n, value, lanes):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        dtype = 'int8'
        ctx = tvm.gpu(0)
        A = te.compute((n, lanes), lambda i,j: tvm.tir.const(value, dtype=dtype))
        s = te.create_schedule(A.op)
        y, x = s[A].op.axis
        s[A].vectorize(x)
        s[A].bind(y, bx)
        fun = tvm.build(s, [A], "cuda", name="make_int8x4")
        np_a = np.full((n, lanes), value, dtype=dtype)
        a = tvm.nd.empty(np_a.shape, dtype, ctx)
        fun(a)
        np.testing.assert_equal(a.asnumpy(), np_a)
    check_cuda(64, 0xAB, 4)
    check_cuda(64, 0, 4)
    check_cuda(64, -3, 4)
    check_cuda(64, 0xAB, 3)
    check_cuda(64, 0, 3)
    check_cuda(64, -3, 3)
    check_cuda(64, 0xAB, 2)
    check_cuda(64, 0, 2)
    check_cuda(64, -3, 2)


def test_cuda_inf_nan():
    target = 'cuda'
    def check_inf_nan(ctx, n, value, dtype):
        A = te.placeholder((n,), name='A', dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name='C')
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tx)
        fun = tvm.build(s, [A, C], target)
        a = tvm.nd.empty((n,), A.dtype, ctx)
        c = tvm.nd.empty((n,), A.dtype, ctx)
        # Only need to test compiling here
        fun(a, c)

    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return

    ctx = tvm.context(target, 0)

    check_inf_nan(ctx, 1, -float('inf'), 'float32')
    check_inf_nan(ctx, 1, -float('inf'), 'float64')
    check_inf_nan(ctx, 1, float('inf'), 'float32')
    check_inf_nan(ctx, 1, float('inf'), 'float64')
    check_inf_nan(ctx, 1, float('nan'), 'float32')
    check_inf_nan(ctx, 1, float('nan'), 'float64')


def test_cuda_shuffle():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return

    idxm = tvm.tir.indexmod
    a = te.placeholder((64, ), 'int32')
    b = te.placeholder((64, ), 'int32')
    c = te.compute((64, ), lambda x: a[x] + b[x - idxm(x, 4) + (3 - idxm(x, 4))])
    sch = te.create_schedule(c.op)
    x = c.op.axis[0]
    xo, xi = sch[c].split(x, 4)
    thrx = te.thread_axis("threadIdx.x")
    sch[c].bind(xo, thrx)
    sch[c].vectorize(xi)

    def MyVectorize():
        def vectorizer(op):
            if op.for_type == tvm.tir.For.Vectorized:
                four = tvm.tir.const(4, 'int32')
                idx = tvm.tir.Ramp(thrx.var * four, tvm.tir.const(1, 'int32'), 4)
                all_ones = tvm.tir.const(1, 'int32x4')
                store = op.body
                value = store.value
                new_a = tvm.tir.Load('int32x4', value.a.buffer_var, idx, all_ones)
                bs, ids = [], []
                for i in range(4):
                    bs.append(tvm.tir.Load('int32', value.b.buffer_var, thrx.var * four + tvm.tir.const(i, 'int32')))
                    ids.append(tvm.tir.const(3 - i, 'int32'))
                new_b = tvm.tir.Shuffle(bs, ids)
                return tvm.tir.Store(store.buffer_var, new_a + new_b, idx, all_ones)
            return None

        def _transform(f, *_):
            return f.with_body(
                tvm.tir.stmt_functor.ir_transform(f.body, None, vectorizer, ['For']))
        return tvm.tir.transform.prim_func_pass(_transform, opt_level=0, name="MyVectorize")

    with tvm.target.build_config(add_lower_pass=[(1, MyVectorize())]):
        module = tvm.build(sch, [a, b, c], target='cuda')
        a_ = np.array(list(range(64)), dtype='int32')
        b_ = np.array((list(range(4))[::-1]) * 16, dtype='int32')
        c_ = np.zeros((64, ), dtype='int32')
        ref = a_ +  np.array((list(range(4))) * 16, dtype='int32')
        nda, ndb, ndc = [tvm.nd.array(i, tvm.gpu(0)) for i in [a_, b_, c_]]
        module(nda, ndb, ndc)
        tvm.testing.assert_allclose(ndc.asnumpy(), ref)


def test_cuda_reducition_binding():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return

    k = te.reduce_axis((0, 32), 'k')
    A = te.placeholder((96, 32), name='A')
    B = te.compute( (96,), lambda m:
                     te.sum(A[m, k], axis=k),
                     name='B')
    s = te.create_schedule(B.op)

    s[B].reorder(B.op.reduce_axis[0], B.op.axis[0])

    mo, _ = s[B].split(B.op.axis[0], 32)
    s[B].bind(mo, te.thread_axis("blockIdx.x"))

    fcuda = tvm.build(s, [A, B], "cuda")

def test_rfactor_predicates():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return

    n = te.reduce_axis((0, 129), 'n')
    A = te.placeholder((129,), name='A')
    B = te.compute( (1, ), lambda b:
                     te.sum(A[n],
                             axis=n),
                     name='B'
    )

    s = te.create_schedule(B.op)

    _, ni = s[B].split(s[B].op.reduce_axis[0], factor=8)

    BF = s.rfactor(B, ni, 0)
    s[B].set_store_predicate(tx.var.equal(0))

    s[B].bind(s[B].op.reduce_axis[0], tx)
    s[B].bind(s[B].op.axis[0], bx)

    s[BF].compute_at(s[B], s[B].op.axis[0])

    _, noi = s[BF].split(s[BF].op.reduce_axis[0], factor=2)

    BF2 = s.rfactor(BF, noi, 0)

    s[BF].bind(s[BF].op.axis[0], tx)
    s[BF2].compute_at(s[BF], s[BF].op.axis[1])

    fcuda = tvm.build(s, [A, B], "cuda")


@unittest.skipIf(not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"), "skip because cuda is not enabled..")
def test_cuda_const_float_to_half():
    # This import is required to use nvcc to perform code gen;
    # otherwise it is found that the code gen is done by nvrtc.
    from tvm import autotvm
    shape = (2, 3, 4)
    a = te.placeholder(shape, dtype='float16', name='a')
    b = tvm.tir.const(0.5, dtype='float16')
    c = te.compute(shape, lambda i, j, k: a[i, j, k] > b, name='c')
    s = te.create_schedule(c.op)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    bx, tx = s[c].split(fused, factor=64)
    s[c].bind(bx, te.thread_axis('blockIdx.x'))
    s[c].bind(tx, te.thread_axis('threadIdx.x'))

    func = tvm.build(s, [a, c], 'cuda')
    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=shape).astype(a.dtype)
    c_np = np.zeros(shape=shape, dtype=c.dtype)
    a = tvm.nd.array(a_np, ctx)
    c = tvm.nd.array(c_np, ctx)
    func(a, c)
    np.testing.assert_equal(c.asnumpy(), a_np > b.value)

def test_cuda_reduction():
    def check_cuda(dtype, m=32, n=32):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        a = te.placeholder((m, n), name="a", dtype=dtype)
        b = te.placeholder((m, n), name="b", dtype=dtype)
        c = a + b
        d = a * b
        e = topi.elemwise_sum([c, d])
        g = topi.sum(e)
        with tvm.target.cuda():
            sg = topi.cuda.schedule_reduce(g)
            ctx = tvm.gpu(0)
            func = tvm.build(sg, [a, b, g], 'cuda')
            a_np = np.random.uniform(size=(m, n)).astype(a.dtype)
            b_np = np.random.uniform(size=(m, n)).astype(b.dtype)
            g_np = np.sum(np.add(a_np * b_np, a_np + b_np))
            a_nd = tvm.nd.array(a_np, ctx)
            b_nd = tvm.nd.array(b_np, ctx)
            g_nd = tvm.nd.array(np.zeros(g_np.shape, dtype=g_np.dtype), ctx)
            func(a_nd, b_nd, g_nd)
            tvm.testing.assert_allclose(g_nd.asnumpy(), g_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")

def test_cuda_mix_threaded_and_normal_reduction():
    def check_cuda(dtype, m=32, n=32):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        a = tvm.te.placeholder((m, n), name="a", dtype=dtype)
        b = topi.sum(a)
        with tvm.target.cuda():
            sb = tvm.te.create_schedule(b.op)
            i, _ = b.op.reduce_axis
            sb[b].bind(i, tvm.te.thread_axis("threadIdx.x"))
            ctx = tvm.gpu(0)
            func = tvm.build(sb, [a, b], 'cuda')
            a_np = np.random.uniform(size=(m, n)).astype(a.dtype)
            b_np = np.sum(a_np)
            a_nd = tvm.nd.array(a_np, ctx)
            b_nd = tvm.nd.array(np.zeros(b_np.shape, dtype=b_np.dtype), ctx)
            func(a_nd, b_nd)
            tvm.testing.assert_allclose(b_nd.asnumpy(), b_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")

def test_cuda_floordiv_with_vectorization():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return

    with tvm.target.cuda():
        # B[i] = A[floordiv(i, k)]
        n = 256
        k = 37
        A = te.placeholder((n,), name='A')
        B = te.compute((n,), lambda i: A[tvm.tir.floordiv(i, k)], name='B')
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], nparts=1)
        xio, xii = s[B].split(xi, factor=4)
        s[B].vectorize(xii)
        s[B].bind(xo, bx)
        s[B].bind(xio, tx)
        func = tvm.build(s, [A, B], 'cuda')

        ctx = tvm.gpu(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        b_np = np.array([a_np[i//k] for i in range(0, n)])
        a_nd = tvm.nd.array(a_np, ctx)
        b_nd = tvm.nd.array(np.zeros(b_np.shape, dtype=b_np.dtype), ctx)
        func(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.asnumpy(), b_np, rtol=1e-3)

def test_vectorized_casts():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return

    def check(t0, t1):
        if (t0 ==  "float16" or t1 == "float16") and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        # compute
        n = 128
        A = te.placeholder((n,), dtype=t0, name='A')
        B = te.placeholder((n,), dtype=t1, name='B')
        C = te.compute((n,), lambda i: A[i] + topi.cast(B[i], A.dtype), name='C')

        # schedule
        s = tvm.te.create_schedule(C.op)
        ob, ib = s[C].split(s[C].op.axis[0], nparts=32)
        _, iib = s[C].split(ib, factor=4)
        s[C].vectorize(iib)
        s[C].bind(ob, tx)
        func = tvm.build(s, [A, B, C], "cuda")

        # correctness
        ctx = tvm.gpu(0)
        low, high = (0, 20) if t0.startswith('u') or t1.startswith('u') else (-10, 10)
        a_np = np.random.randint(low, high, size=n).astype(A.dtype)
        b_np = np.random.randint(low, high, size=n).astype(B.dtype)
        c_np = (a_np + b_np).astype(A.dtype)
        a_nd = tvm.nd.array(a_np, ctx)
        b_nd = tvm.nd.array(b_np, ctx)
        c_nd = tvm.nd.array(np.zeros(c_np.shape, dtype=c_np.dtype), ctx)
        func(a_nd, b_nd, c_nd)
        tvm.testing.assert_allclose(c_nd.asnumpy(), c_np, rtol=1e-3)

    def skip(t0, t1):
        if t0 == t1:
            return True
        # CUDA does support cast between {u}int8 and fp16.
        skip_set = {"float16", "uint8", "int8"}
        if t0 in skip_set and t1 in skip_set:
            return True
        return False

    types = ["float16", "float32", "int8", "uint8", "int16", "uint16", "int32", "uint32"]
    for t0, t1 in [(x, y) for x in types for y in types if not skip(x, y)]:
        check(t0, t1)

def sched(B):
    s = te.create_schedule(B.op)
    io, ii = s[B].split(s[B].op.axis[0], nparts=1)
    iio, iii = s[B].split(ii, nparts=32)
    _, iiii = s[B].split(iii, factor=4)
    s[B].vectorize(iiii)
    s[B].bind(io, bx)
    s[B].bind(iio, tx)
    return s

def test_vectorized_intrin1():
    test_funcs = [
        (tvm.tir.floor, lambda x : np.floor(x)),
        (tvm.tir.ceil,  lambda x : np.ceil(x)),
        (tvm.tir.trunc, lambda x : np.trunc(x)),
        (tvm.tir.abs,   lambda x : np.fabs(x)),
        (tvm.tir.round, lambda x : np.round(x)),
        (tvm.tir.exp,   lambda x : np.exp(x)),
        (tvm.tir.exp2,  lambda x : np.exp2(x)),
        (tvm.tir.exp10, lambda x : np.power(10,x)),
        (tvm.tir.log,   lambda x : np.log(x)),
        (tvm.tir.log2,  lambda x : np.log2(x)),
        (tvm.tir.log10, lambda x : np.log10(x)),
        (tvm.tir.tan,   lambda x : np.tan(x)),
        (tvm.tir.cos,   lambda x : np.cos(x)),
        (tvm.tir.cosh,  lambda x : np.cosh(x)),
        (tvm.tir.sin,   lambda x : np.sin(x)),
        (tvm.tir.sinh,  lambda x : np.sinh(x)),
        (tvm.tir.atan,  lambda x : np.arctan(x)),
        (tvm.tir.tanh,  lambda x : np.tanh(x)),
        (tvm.tir.sqrt,  lambda x : np.sqrt(x)),
    ]
    def run_test(tvm_intrin, np_func, dtype):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        # set of intrinsics does not support fp16 yet.
        skip_set = {tvm.tir.abs,
                    tvm.tir.round,
                    tvm.tir.tan,
                    tvm.tir.atan,
                    tvm.tir.tanh,
                    tvm.tir.cosh,
                    tvm.tir.sinh}
        if dtype == "float16" and tvm_intrin in skip_set:
            print("Skip because '{0}' does not support fp16 yet".format(tvm_intrin.__name__))
            return

        n = 128
        A = te.placeholder((n,), dtype=dtype, name='A')
        B = te.compute((n,), lambda *i: tvm_intrin(A(*i)), name='B')
        s = sched(B)
        f = tvm.build(s, [A, B], "cuda")
        ctx = tvm.gpu(0)
        a = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(shape=(n,)).astype(A.dtype), ctx)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), np_func(a.asnumpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func, "float32")
        run_test(*func, "float16")

def test_vectorized_intrin2(dtype="float32"):
    c2 = tvm.tir.const(2, dtype=dtype)
    test_funcs = [
        (tvm.tir.power, lambda x : np.power(x, 2.0)),
        (tvm.tir.fmod,  lambda x : np.fmod(x, 2.0))
    ]
    def run_test(tvm_intrin, np_func):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return

        n = 128
        A = te.placeholder((n,), dtype=dtype, name='A')
        B = te.compute((n,), lambda i: tvm_intrin(A[i], c2), name='B')
        s = sched(B)
        f = tvm.build(s, [A, B], "cuda")
        ctx = tvm.gpu(0)
        a = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(shape=(n,)).astype(A.dtype), ctx)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), np_func(a.asnumpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func)

def test_vectorized_popcount():
    def ref_popcount(x):
        cnt = 0
        while x:
            x -= x & -x
            cnt += 1
        return cnt

    def run_test(dtype):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return

        n = 128
        A = te.placeholder((n,), dtype=dtype, name='A')
        B = te.compute((n,), lambda i: tvm.tir.popcount(A[i]), name='B')
        s = sched(B)
        f = tvm.build(s, [A, B], "cuda")
        ctx = tvm.gpu(0)
        a = tvm.nd.array(np.random.randint(0, 100000, size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(shape=(n,)).astype(B.dtype), ctx)
        f(a, b)
        ref = np.vectorize(ref_popcount)(a.asnumpy())
        tvm.testing.assert_allclose(b.asnumpy(), ref)

    run_test("uint32")
    run_test("uint64")

def test_cuda_vectorize_load_permute_pad():
    def check_cuda(dtype, n, l, padding, lanes):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        ctx = tvm.gpu(0)
        A = tvm.te.placeholder((n, l), name='A', dtype=dtype)
        B = tvm.te.compute((n // lanes, l + 2 * padding, lanes),
                           lambda i, j, k: tvm.te.if_then_else(
            tvm.te.any(j < padding, j >= l + padding),
            tvm.runtime.convert(0).astype(dtype), A[i * lanes + k, j - padding]),
            name='B')
        s = te.create_schedule(B.op)
        block, thread, vectorize = s[B].op.axis
        s[B].bind(block, bx)
        s[B].bind(thread, tx)
        s[B].vectorize(vectorize)
        fun = tvm.build(s, [A, B], "cuda", name="vector_load_permute_pad")
        np_a = np.random.randint(
            low=-128, high=127, size=(n, l)).astype(A.dtype)
        a = tvm.nd.empty((n, l), A.dtype, ctx).copyfrom(np_a)
        b = tvm.nd.empty((n // lanes, l + padding * 2, lanes), B.dtype, ctx)
        fun(a, b)
        np_a_reshape = np_a.reshape(n // lanes, lanes, l).transpose(0, 2, 1)
        ref = np.pad(np_a_reshape, ((0, 0), (padding, padding),
                                    (0, 0)), mode='constant', constant_values=0)
        tvm.testing.assert_allclose(b.asnumpy(), ref)

    check_cuda("int8", 64, 16, 3, 2)
    check_cuda("uint8", 64, 16, 3, 2)
    check_cuda("int8", 64, 16, 3, 4)
    check_cuda("uint8", 64, 16, 3, 4)
    check_cuda("int32", 64, 16, 3, 4)
    check_cuda("float16", 64, 16, 3, 4)
    check_cuda("float32", 64, 16, 3, 4)

if __name__ == "__main__":
    test_cuda_vectorize_add()
    test_cuda_multiply_add()
    test_cuda_vectorize_load()
    test_cuda_make_int8()
    test_cuda_inf_nan()
    test_cuda_shuffle()
    test_vectorized_casts()
    test_cuda_reducition_binding()
    test_rfactor_predicates()
    test_cuda_const_float_to_half()
    test_cuda_reduction()
    test_cuda_mix_threaded_and_normal_reduction()
    test_cuda_floordiv_with_vectorization()
    test_vectorized_intrin1()
    test_vectorized_intrin2()
    test_vectorized_popcount()
    test_cuda_vectorize_load_permute_pad()
