
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
import numpy as np
import topi
import unittest
from tvm.contrib.nvcc import have_fp16, have_int8
from tvm.contrib import nvcc

tx = tvm.thread_axis("threadIdx.x")
bx = tvm.thread_axis("blockIdx.x")

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
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.compute((n,), lambda i: A[i] + tvm.const(1, A.dtype), name='B')
        s = tvm.create_schedule(B.op)
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
    check_cuda("int8",    64, 4)
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
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.placeholder((n,), name='B', dtype="%sx%d" % (dtype, lanes))
        C = tvm.placeholder((n,), name='C', dtype="int32")
        D = tvm.compute((n,),
                        lambda i: tvm.call_pure_extern("int32", "__dp4a", A[i], B[i], C[i]), name='D')
        s = tvm.create_schedule(D.op)
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
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.compute((n,), lambda i: A[i], name='B')
        s = tvm.create_schedule(B.op)
        block, thread = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(block, bx)
        s[B].bind(thread, tx)
        fun = tvm.build(s, [A, B], "cuda", name="vector_load")
        np_a = np.random.randint(low=-128, high=127, size=(n,lanes))
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a,b)
        tvm.testing.assert_allclose(a.asnumpy(), b.asnumpy())
    check_cuda("int8", 64, 8)
    check_cuda("int8", 64, 16)

def test_cuda_make_int8x4():
    def check_cuda(n, value):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        lanes = 4
        dtype = 'int8'
        ctx = tvm.gpu(0)
        A = tvm.compute((n, lanes), lambda i,j: tvm.const(value, dtype=dtype))
        s = tvm.create_schedule(A.op)
        y, x = s[A].op.axis
        s[A].vectorize(x)
        s[A].bind(y, bx)
        fun = tvm.build(s, [A], "cuda", name="make_int8x4")
        np_a = np.full((n, lanes), value, dtype=dtype)
        a = tvm.nd.empty(np_a.shape, dtype, ctx)
        fun(a)
        np.testing.assert_equal(a.asnumpy(), np_a)
    check_cuda(64, 0xAB)
    check_cuda(64, 0)
    check_cuda(64, -3)


def test_cuda_inf_nan():
    target = 'cuda'
    def check_inf_nan(ctx, n, value, dtype):
        A = tvm.placeholder((n,), name='A', dtype=dtype)
        inf_value = tvm.const(value, dtype=dtype)
        C = tvm.compute((n,), lambda i: inf_value, name='C')
        s = tvm.create_schedule(C.op)
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

    idxm = tvm.indexmod
    a = tvm.placeholder((64, ), 'int32')
    b = tvm.placeholder((64, ), 'int32')
    c = tvm.compute((64, ), lambda x: a[x] + b[x - idxm(x, 4) + (3 - idxm(x, 4))])
    sch = tvm.create_schedule(c.op)
    x = c.op.axis[0]
    xo, xi = sch[c].split(x, 4)
    thrx = tvm.thread_axis("threadIdx.x")
    sch[c].bind(xo, thrx)
    sch[c].vectorize(xi)

    def my_vectorize(stmt):
        def vectorizer(op):
            if op.for_type == tvm.tir.For.Vectorized:
                four = tvm.const(4, 'int32')
                idx = tvm.tir.Ramp(thrx.var * four, tvm.const(1, 'int32'), 4)
                all_ones = tvm.const(1, 'int32x4')
                store = op.body
                value = store.value
                new_a = tvm.tir.Load('int32x4', value.a.buffer_var, idx, all_ones)
                bs, ids = [], []
                for i in range(4):
                    bs.append(tvm.tir.Load('int32', value.b.buffer_var, thrx.var * four + tvm.const(i, 'int32')))
                    ids.append(tvm.const(3 - i, 'int32'))
                new_b = tvm.tir.Shuffle(bs, ids)
                return tvm.tir.Store(store.buffer_var, new_a + new_b, idx, all_ones)
            return None
        return tvm.ir_pass.IRTransform(stmt, None, vectorizer, ['For'])

    with tvm.build_config(add_lower_pass=[(1, my_vectorize)]):
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

    k = tvm.reduce_axis((0, 32), 'k')
    A = tvm.placeholder((96, 32), name='A')
    B = tvm.compute( (96,), lambda m:
                     tvm.sum(A[m, k], axis=k),
                     name='B')
    s = tvm.create_schedule(B.op)

    s[B].reorder(B.op.reduce_axis[0], B.op.axis[0])

    mo, _ = s[B].split(B.op.axis[0], 32)
    s[B].bind(mo, tvm.thread_axis("blockIdx.x"))

    fcuda = tvm.build(s, [A, B], "cuda")

def test_rfactor_predicates():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return

    n = tvm.reduce_axis((0, 129), 'n')
    A = tvm.placeholder((129,), name='A')
    B = tvm.compute( (1, ), lambda b:
                     tvm.sum(A[n],
                             axis=n),
                     name='B'
    )

    s = tvm.create_schedule(B.op)

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
    a = tvm.placeholder(shape, dtype='float16', name='a')
    b = tvm.const(0.5, dtype='float16')
    c = tvm.compute(shape, lambda i, j, k: a[i, j, k] > b, name='c')
    s = tvm.create_schedule(c.op)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    bx, tx = s[c].split(fused, factor=64)
    s[c].bind(bx, tvm.thread_axis('blockIdx.x'))
    s[c].bind(tx, tvm.thread_axis('threadIdx.x'))

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

        a = tvm.placeholder((m, n), name="a", dtype=dtype)
        b = tvm.placeholder((m, n), name="b", dtype=dtype)
        c = a + b
        d = a * b
        e = topi.elemwise_sum([c, d])
        g = topi.sum(e)
        with tvm.target.cuda():
            sg = topi.generic.schedule_reduce(g)
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

if __name__ == "__main__":
    test_cuda_vectorize_add()
    test_cuda_multiply_add()
    test_cuda_vectorize_load()
    test_cuda_make_int8x4()
    test_cuda_inf_nan()
    test_cuda_shuffle()
    test_cuda_reducition_binding()
    test_rfactor_predicates()
    test_cuda_const_float_to_half()
    test_cuda_reduction()