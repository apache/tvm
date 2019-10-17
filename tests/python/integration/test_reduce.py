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


def test_reduce_prims():
    def test_prim(reducer, np_reducer):
        # graph
        n = tvm.var('n')
        m = tvm.var('m')
        A = tvm.placeholder((n, m), name='A')
        R = tvm.compute((n, ), lambda i: tvm.expr.Select((i > 1), 1, 0), name='R')
        k = tvm.reduce_axis((0, m))
        B = tvm.compute((n,), lambda i: reducer(A[i, k], axis=k, where=(R[i]==1)), name='B')
        # schedule
        s = tvm.create_schedule(B.op)
        # create iter var and assign them tags.
        num_thread = 1
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
        s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
        s[R].compute_inline()

        # one line to build the function.
        def check_device(device, host="llvm"):
            ctx = tvm.context(device, 0)
            if not tvm.module.enabled(host):
                return
            if not ctx.exist:
                print("skip because %s is not enabled.." % device)
                return
            freduce = tvm.build(s,
                             args=[A, B],
                             target=device, target_host=host,
                             name="myreduce")
            # launch the kernel.
            n = 1028
            m = 129
            x = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
            y = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
            freduce(x, y)
            npy = y.asnumpy()
            npy[:2] = 0
            res = np_reducer(x.asnumpy(), axis=1)
            res[:2] = 0
            tvm.testing.assert_allclose(npy, res, rtol=1e-4)

        check_device("metal")
        check_device("vulkan")
        check_device("cuda")
        check_device("opencl")
    test_prim(tvm.sum, np.sum)
    test_prim(tvm.min, np.amin)
    test_prim(tvm.max, np.amax)


def test_rfactor():
    n = tvm.convert(1027)
    A = tvm.placeholder((n,), name='A')
    k = tvm.reduce_axis((0, n))
    B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')
    # schedule
    s = tvm.create_schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf)
    s[BF].parallel(BF.op.axis[0])
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.module.enabled(target):
            return
        ctx = tvm.cpu(0)
        fapi = tvm.lower(s, args=[A, B])
        fsum = tvm.build(fapi,
                         target=target,
                         name="mysum")
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
        b  = tvm.nd.array(np.zeros(1, dtype=B.dtype), ctx)
        fsum(a, b)
        res = np.sum(a.asnumpy(), axis=0)
        tvm.testing.assert_allclose(
            b.asnumpy(), res, rtol=1e-4)

    check_target()

def test_rfactor_factor_axis():
    n = tvm.convert(1027)
    A = tvm.placeholder((n,), name='A')
    k = tvm.reduce_axis((0, n))
    B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')
    # schedule
    s = tvm.create_schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf, 1)
    s[BF].parallel(BF.op.axis[0])
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.module.enabled(target):
            return
        ctx = tvm.cpu(0)
        fapi = tvm.lower(s, args=[A, B])
        fsum = tvm.build(fapi,
                         target=target,
                         name="mysum")
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
        b  = tvm.nd.array(np.zeros(1, dtype=B.dtype), ctx)
        fsum(a, b)
        res = np.sum(a.asnumpy(), axis=0)
        tvm.testing.assert_allclose(
            b.asnumpy(), res, rtol=1e-4)

    check_target()


def test_rfactor_threads():
    nn = 1027
    mm = 10
    n = tvm.convert(nn)
    m = tvm.convert(mm)
    A = tvm.placeholder((m, n), name='A')
    k = tvm.reduce_axis((0, n))
    nthread = 16
    B = tvm.compute((m,), lambda i: tvm.sum(A[i, k], axis=k, where=(i>1)), name='B')
    # schedule
    s = tvm.create_schedule(B.op)
    ko, kf = s[B].split(k, factor=nthread)
    BF = s.rfactor(B, kf)
    bx, ty = s[B].split(s[B].op.axis[0], factor=nthread)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(ty, tvm.thread_axis("threadIdx.y"))
    tx = s[B].op.reduce_axis[0]
    thread_x = tvm.thread_axis("threadIdx.x")
    s[B].bind(tx, thread_x)
    s[BF].compute_at(s[B], tx)
    s[B].set_store_predicate(thread_x.var.equal(0))

    # one line to build the function.
    def check_target(device, host="stackvm"):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("skip because %s is not enabled.." % device)
            return

        fapi = tvm.lower(s, args=[A, B])
        fsum = tvm.build(fapi,
                         target=device,
                         name="mysum")
        # launch the kernel.
        n = nn
        m = mm
        a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(A.dtype), ctx)
        b  = tvm.nd.array(np.zeros(m, dtype=B.dtype), ctx)
        fsum(a, b)
        res = np.sum(a.asnumpy(), axis=1)
        res[:2] = 0
        tvm.testing.assert_allclose(
            b.asnumpy(), res, rtol=1e-4)

    check_target("vulkan")
    check_target("cuda")
    check_target("metal")
    check_target("opencl")


def test_rfactor_elemwise_threads():
    n = 1025
    m = 10
    A = tvm.placeholder((m, n), name='A')
    k = tvm.reduce_axis((0, n))
    nthread = 16
    B = tvm.compute((m,), lambda i: tvm.sum(A[i, k], axis=k), name='B')
    BB = tvm.compute((m,), lambda i: B[i] + 1, name='BB')
    C = tvm.compute((m,), lambda i: BB[i] + 1, name='C')
    # schedule
    s = tvm.create_schedule(C.op)
    s[BB].compute_inline()
    bx, ty = s[C].split(s[C].op.axis[0], factor=nthread)
    ko, kf = s[B].split(k, factor=nthread)
    BF = s.rfactor(B, kf)
    s[B].compute_at(s[C], ty)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    tx = s[B].op.reduce_axis[0]
    thread_x = tvm.thread_axis("threadIdx.x")
    s[B].bind(tx, thread_x)
    s[BF].compute_at(s[B], tx)
    # Since thread_x is shared across reductions
    # only one of them need to do write back
    s[B].set_store_predicate(thread_x.var.equal(0))
    s[C].set_store_predicate(thread_x.var.equal(0))

    # one line to build the function.
    def check_target(device, host="stackvm"):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("skip because %s is not enabled.." % device)
            return
        fapi = tvm.lower(s, args=[A, C])
        fsum = tvm.build(fapi,
                         target=device,
                         name="mysum")
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(A.dtype), ctx)
        b  = tvm.nd.array(np.zeros(m, dtype=B.dtype), ctx)
        fsum(a, b)
        res = np.sum(a.asnumpy(), axis=1) + 2
        tvm.testing.assert_allclose(
            b.asnumpy(), res, rtol=1e-4)

    check_target("vulkan")
    check_target("cuda")
    check_target("metal")
    check_target("opencl")

def test_argmax():
    def fcombine(x, y):
        lhs = tvm.make.Select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm.make.Select((x[1] >= y[1]), x[1], y[1])
        return lhs, rhs

    def fidentity(t0, t1):
        return tvm.const(-1, t0), tvm.min_value(t1)

    argmax = tvm.comm_reducer(fcombine,
                              fidentity,
                              name='argmax')
    m = tvm.var('m')
    n = tvm.var('n')
    idx = tvm.placeholder((m, n), name='idx', dtype='int32')
    val = tvm.placeholder((m, n), name='val', dtype='float32')
    k = tvm.reduce_axis((0, n), 'k')
    T0, T1 = tvm.compute((m,), lambda i: argmax((idx[i,k], val[i,k]), axis=k), name='T')
    s = tvm.create_schedule(T0.op)

    def check_target():
        device = 'cpu'
        if not tvm.module.enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        ctx = tvm.context(device, 0)
        fapi = tvm.lower(s, args=[idx, val, T0, T1])
        fargmax = tvm.build(fapi,
                            target='llvm',
                            name="argmax")

        mm = 12
        nn = 16
        np_idx = np.repeat(np.arange(nn, dtype='int32').reshape(1, nn), mm, axis=0)
        np_val = np.random.uniform(size=(mm, nn)).astype('float32')
        np_res = np.argmax(np_val, axis=1)

        nd_idx  = tvm.nd.array(np_idx, ctx)
        nd_val  = tvm.nd.array(np_val, ctx)
        nd_res0 = tvm.nd.array(np.zeros(mm, dtype='int32'), ctx)
        nd_res1 = tvm.nd.array(np.zeros(mm, dtype='float32'), ctx)
        fargmax(nd_idx, nd_val, nd_res0, nd_res1)
        tvm.testing.assert_allclose(np_res, nd_res0.asnumpy())

    check_target()


def test_rfactor_argmax():
    def fcombine(x, y):
        lhs = tvm.make.Select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm.make.Select((x[1] >= y[1]), x[1], y[1])
        return lhs, rhs

    def fidentity(t0, t1):
        return tvm.const(-1, t0), tvm.min_value(t1)

    argmax = tvm.comm_reducer(fcombine,
                              fidentity,
                              name='argmax')

    nn = 1027
    mm = 10
    n = tvm.convert(nn)
    m = tvm.convert(mm)
    A0 = tvm.placeholder((m, n), name='A0', dtype='int32')
    A1 = tvm.placeholder((m, n), name='A1', dtype='float32')
    k = tvm.reduce_axis((0, n))
    B0, B1 = tvm.compute((m,), lambda i: argmax((A0[i, k], A1[i, k]), axis=k), name='B')

    # schedule
    s = tvm.create_schedule(B0.op)
    nthread = 16
    ko, kf = s[B0].split(k, factor=nthread)
    BF0, BF1 = s.rfactor(B0, kf)
    bx, ty = s[B0].split(s[B0].op.axis[0], factor=nthread)
    s[B0].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B0].bind(ty, tvm.thread_axis("threadIdx.y"))
    tx = s[B0].op.reduce_axis[0]
    thread_x = tvm.thread_axis("threadIdx.x")
    s[B0].bind(tx, thread_x)
    s[BF0.op].compute_at(s[B0], tx)
    s[B0].set_store_predicate(thread_x.var.equal(0))

    def check_target(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("skip because %s is not enabled.." % device)
            return
        fapi = tvm.lower(s, args=[A0, A1, B0, B1])
        fargmax = tvm.build(fapi,
                            target=device,
                            name="argmax")

        np_idx = np.repeat(np.arange(nn, dtype='int32').reshape(1, nn), mm, axis=0)
        np_val = np.random.uniform(size=(mm, nn)).astype('float32')
        np_res = np.argmax(np_val, axis=1)

        nd_idx  = tvm.nd.array(np_idx, ctx)
        nd_val  = tvm.nd.array(np_val, ctx)
        nd_res0 = tvm.nd.array(np.zeros(mm, dtype='int32'), ctx)
        nd_res1 = tvm.nd.array(np.zeros(mm, dtype='float32'), ctx)
        fargmax(nd_idx, nd_val, nd_res0, nd_res1)
        tvm.testing.assert_allclose(np_res, nd_res0.asnumpy())

    check_target("cuda")
    check_target("vulkan")

if __name__ == "__main__":
    test_rfactor_elemwise_threads()
    test_rfactor_threads()
    test_rfactor_factor_axis()
    test_rfactor()
    test_reduce_prims()
    test_argmax()
    test_rfactor_argmax()
