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
import pytest
import numpy as np

import tvm
from tvm import te, topi
from tvm.driver.build_module import schedule_to_primfunc
import tvm.testing
import tvm.topi.testing


@tvm.testing.requires_gpu
def test_reduce_prims():
    def test_prim(reducer, np_reducer):
        # graph
        n = tvm.te.size_var("n")
        m = tvm.te.size_var("m")
        A = te.placeholder((n, m), name="A")
        R = te.compute((n,), lambda i: tvm.tir.Select((i > 1), 1, 0), name="R")
        k = te.reduce_axis((0, m))
        B = te.compute((n,), lambda i: reducer(A[i, k], axis=k, where=(R[i] == 1)), name="B")
        # schedule
        s = te.create_schedule(B.op)
        # create iter var and assign them tags.
        num_thread = 1
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, te.thread_axis("blockIdx.x"))
        s[B].bind(xi, te.thread_axis("threadIdx.x"))
        s[R].compute_inline()

        # one line to build the function.
        def check_device(device, host="llvm"):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            freduce = tvm.build(
                s, args=[A, B], target=tvm.target.Target(device, host), name="myreduce"
            )
            # launch the kernel.
            n = 1028
            m = 129
            x = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), dev)
            y = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
            freduce(x, y)
            npy = y.numpy()
            npy[:2] = 0
            res = np_reducer(x.numpy(), axis=1)
            res[:2] = 0
            tvm.testing.assert_allclose(npy, res, rtol=1e-4)

        check_device("metal")
        check_device("vulkan")
        check_device("cuda")
        check_device("opencl")
        check_device("rocm")

    test_prim(te.sum, np.sum)
    test_prim(tvm.te.min, np.amin)
    test_prim(tvm.te.max, np.amax)


def test_init_imm():
    n = tvm.runtime.convert(1027)
    A = te.placeholder((n,), name="A")
    k = te.reduce_axis((0, n))
    B = te.compute((), lambda: te.sum(A[k], axis=k, init=10.0), name="B")
    # schedule
    s = te.create_schedule(B.op)
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.runtime.enabled(target):
            return
        dev = tvm.cpu(0)
        fapi = tvm.lower(s, args=[A, B])
        fsum = tvm.build(fapi, target=target, name="mysum")
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros((), dtype=B.dtype), dev)
        fsum(a, b)
        res = 10.0 + np.sum(a.numpy(), axis=0)
        tvm.testing.assert_allclose(b.numpy(), res, rtol=1e-4)

    check_target()


def test_init():
    n = tvm.runtime.convert(1027)
    A = te.placeholder((n, n), name="A")
    C = te.placeholder((n, n), name="C")
    I = te.placeholder((n, n), name="I")
    k = te.reduce_axis((0, n))
    B = te.compute((n, n), lambda i, j: te.sum(A[i, k] * C[k, j], axis=k, init=I[i, j]), name="B")

    # schedule
    s = te.create_schedule(B.op)
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.runtime.enabled(target):
            return
        dev = tvm.cpu(0)
        fapi = tvm.lower(s, args=[A, C, I, B])
        print(fapi)
        mmult = tvm.build(fapi, target=target, name="mmult")
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(A.dtype), dev)
        c = tvm.nd.array(np.random.uniform(size=(n, n)).astype(C.dtype), dev)
        ii = tvm.nd.array(np.random.uniform(size=(n, n)).astype(B.dtype), dev)
        b = tvm.nd.array(np.zeros((n, n), dtype=B.dtype), dev)
        mmult(a, c, ii, b)
        res = ii.numpy() + np.matmul(a.numpy(), c.numpy())
        tvm.testing.assert_allclose(b.numpy(), res, rtol=1e-4)

    check_target()


def test_rfactor():
    n = tvm.runtime.convert(1027)
    A = te.placeholder((n,), name="A")
    k = te.reduce_axis((0, n))
    B = te.compute((), lambda: te.sum(A[k], axis=k), name="B")
    # schedule
    s = te.create_schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf)
    s[BF].parallel(BF.op.axis[0])
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.testing.device_enabled(target):
            return
        dev = tvm.cpu(0)
        fapi = tvm.lower(s, args=[A, B])
        fsum = tvm.build(fapi, target=target, name="mysum")
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros((), dtype=B.dtype), dev)
        fsum(a, b)
        res = np.sum(a.numpy(), axis=0)
        tvm.testing.assert_allclose(b.numpy(), res, rtol=1e-4)

    check_target()


def test_rfactor_init():
    n = tvm.runtime.convert(1027)
    A = te.placeholder((n, n), name="A")
    C = te.placeholder((n, n), name="C")
    I = te.placeholder((n, n), name="I")
    k = te.reduce_axis((0, n))
    B = te.compute((n, n), lambda i, j: te.sum(A[i, k] * C[k, j], axis=k, init=I[i, j]), name="B")

    # schedule
    s = te.create_schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf, 1)
    s[BF].parallel(BF.op.axis[0])
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.runtime.enabled(target):
            return
        dev = tvm.cpu(0)
        fapi = tvm.lower(s, args=[A, C, I, B])
        print(fapi)
        mmult = tvm.build(fapi, target=target, name="mmult")
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(A.dtype), dev)
        c = tvm.nd.array(np.random.uniform(size=(n, n)).astype(C.dtype), dev)
        ii = tvm.nd.array(np.random.uniform(size=(n, n)).astype(B.dtype), dev)
        b = tvm.nd.array(np.zeros((n, n), dtype=B.dtype), dev)
        mmult(a, c, ii, b)
        res = ii.numpy() + np.matmul(a.numpy(), c.numpy())
        tvm.testing.assert_allclose(b.numpy(), res, rtol=1e-4)

    check_target()


def test_rfactor_factor_axis():
    n = tvm.runtime.convert(1027)
    A = te.placeholder((n,), name="A")
    k = te.reduce_axis((0, n))
    B = te.compute((), lambda: te.sum(A[k], axis=k), name="B")
    # schedule
    s = te.create_schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf, 0)
    s[BF].parallel(BF.op.axis[0])
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.testing.device_enabled(target):
            return
        dev = tvm.cpu(0)
        fapi = tvm.lower(s, args=[A, B])
        fsum = tvm.build(fapi, target=target, name="mysum")
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros((), dtype=B.dtype), dev)
        fsum(a, b)
        res = np.sum(a.numpy(), axis=0)
        tvm.testing.assert_allclose(b.numpy(), res, rtol=1e-4)

    check_target()


@tvm.testing.requires_gpu
def test_rfactor_threads():
    nn = 1027
    mm = 10
    n = tvm.runtime.convert(nn)
    m = tvm.runtime.convert(mm)
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n))
    nthread = 16
    B = te.compute((m,), lambda i: te.sum(A[i, k], axis=k, where=(i > 1)), name="B")
    # schedule
    s = te.create_schedule(B.op)
    ko, kf = s[B].split(k, factor=nthread)
    BF = s.rfactor(B, kf)
    bx, ty = s[B].split(s[B].op.axis[0], factor=nthread)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(ty, te.thread_axis("threadIdx.y"))
    tx = s[B].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[B].bind(tx, thread_x)
    s[BF].compute_at(s[B], tx)
    s[B].set_store_predicate(thread_x.var.equal(0))

    # one line to build the function.
    def check_target(device, host="stackvm"):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return

        fapi = tvm.lower(s, args=[A, B])
        fsum = tvm.build(fapi, target=device, name="mysum")
        # launch the kernel.
        n = nn
        m = mm
        a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(m, dtype=B.dtype), dev)
        fsum(a, b)
        res = np.sum(a.numpy(), axis=1)
        res[:2] = 0
        tvm.testing.assert_allclose(b.numpy(), res, rtol=1e-4)

    check_target("vulkan")
    check_target("cuda")
    check_target("metal")
    check_target("opencl")
    check_target("rocm")


@tvm.testing.requires_gpu
def test_rfactor_elemwise_threads():
    n = 1025
    m = 10
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n))
    nthread = 16
    B = te.compute((m,), lambda i: te.sum(A[i, k], axis=k), name="B")
    BB = te.compute((m,), lambda i: B[i] + 1, name="BB")
    C = te.compute((m,), lambda i: BB[i] + 1, name="C")
    # schedule
    s = te.create_schedule(C.op)
    s[BB].compute_inline()
    bx, ty = s[C].split(s[C].op.axis[0], factor=nthread)
    ko, kf = s[B].split(k, factor=nthread)
    BF = s.rfactor(B, kf)
    s[B].compute_at(s[C], ty)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    tx = s[B].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[B].bind(tx, thread_x)
    s[BF].compute_at(s[B], tx)
    # Since thread_x is shared across reductions
    # only one of them need to do write back
    s[B].set_store_predicate(thread_x.var.equal(0))
    s[C].set_store_predicate(thread_x.var.equal(0))

    # one line to build the function.
    def check_target(device, host="stackvm"):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        fapi = tvm.lower(s, args=[A, C])
        fsum = tvm.build(fapi, target=device, name="mysum")
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(m, dtype=B.dtype), dev)
        fsum(a, b)
        res = np.sum(a.numpy(), axis=1) + 2
        tvm.testing.assert_allclose(b.numpy(), res, rtol=1e-4)

    check_target("vulkan")
    check_target("cuda")
    check_target("metal")
    check_target("opencl")
    check_target("rocm")


def test_argmax():
    def fcombine(x, y):
        lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
        return lhs, rhs

    def fidentity(t0, t1):
        return tvm.tir.const(-1, t0), tvm.te.min_value(t1)

    argmax = te.comm_reducer(fcombine, fidentity, name="argmax")
    m = te.size_var("m")
    n = te.size_var("n")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    val = te.placeholder((m, n), name="val", dtype="float32")
    k = te.reduce_axis((0, n), "k")
    T0, T1 = te.compute((m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="T")
    s = te.create_schedule(T0.op)

    def check_target():
        device = "cpu"
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        dev = tvm.device(device, 0)
        fapi = tvm.lower(s, args=[idx, val, T0, T1])
        fargmax = tvm.build(fapi, target="llvm", name="argmax")

        mm = 12
        nn = 16
        np_idx = np.repeat(np.arange(nn, dtype="int32").reshape(1, nn), mm, axis=0)
        np_val = np.random.uniform(size=(mm, nn)).astype("float32")
        np_res = np.argmax(np_val, axis=1)

        nd_idx = tvm.nd.array(np_idx, dev)
        nd_val = tvm.nd.array(np_val, dev)
        nd_res0 = tvm.nd.array(np.zeros(mm, dtype="int32"), dev)
        nd_res1 = tvm.nd.array(np.zeros(mm, dtype="float32"), dev)
        fargmax(nd_idx, nd_val, nd_res0, nd_res1)
        tvm.testing.assert_allclose(np_res, nd_res0.numpy())

    check_target()


@tvm.testing.requires_gpu
def test_rfactor_argmax():
    def fcombine(x, y):
        lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
        return lhs, rhs

    def fidentity(t0, t1):
        return tvm.tir.const(-1, t0), tvm.te.min_value(t1)

    argmax = te.comm_reducer(fcombine, fidentity, name="argmax")

    nn = 1027
    mm = 10
    n = tvm.runtime.convert(nn)
    m = tvm.runtime.convert(mm)
    A0 = te.placeholder((m, n), name="A0", dtype="int32")
    A1 = te.placeholder((m, n), name="A1", dtype="float32")
    k = te.reduce_axis((0, n))
    B0, B1 = te.compute((m,), lambda i: argmax((A0[i, k], A1[i, k]), axis=k), name="B")

    # schedule
    s = te.create_schedule(B0.op)
    nthread = 16
    ko, kf = s[B0].split(k, factor=nthread)
    BF0, BF1 = s.rfactor(B0, kf)
    bx, ty = s[B0].split(s[B0].op.axis[0], factor=nthread)
    s[B0].bind(bx, te.thread_axis("blockIdx.x"))
    s[B0].bind(ty, te.thread_axis("threadIdx.y"))
    tx = s[B0].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[B0].bind(tx, thread_x)
    s[BF0.op].compute_at(s[B0], tx)
    s[B0].set_store_predicate(thread_x.var.equal(0))

    def check_target(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        fapi = tvm.lower(s, args=[A0, A1, B0, B1])
        fargmax = tvm.build(fapi, target=device, name="argmax")

        np_idx = np.repeat(np.arange(nn, dtype="int32").reshape(1, nn), mm, axis=0)
        np_val = np.random.uniform(size=(mm, nn)).astype("float32")
        np_res = np.argmax(np_val, axis=1)

        nd_idx = tvm.nd.array(np_idx, dev)
        nd_val = tvm.nd.array(np_val, dev)
        nd_res0 = tvm.nd.array(np.zeros(mm, dtype="int32"), dev)
        nd_res1 = tvm.nd.array(np.zeros(mm, dtype="float32"), dev)
        fargmax(nd_idx, nd_val, nd_res0, nd_res1)
        tvm.testing.assert_allclose(np_res, nd_res0.numpy())

    check_target("cuda")
    check_target("vulkan")
    check_target("rocm")


@tvm.testing.requires_gpu
def test_warp_reduction1():
    nthx = 32
    nthy = 4
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, nthx), "threadIdx.x")
    thread_y = te.thread_axis((0, nthy), "threadIdx.y")

    def check_target(device, m, n):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return

        # compute
        A = te.placeholder((m, n), name="A")
        k = te.reduce_axis((0, n))
        B = te.compute((m,), lambda i: te.max(A[i][k], axis=k), name="B")
        s = te.create_schedule(B.op)

        # schedule
        k = s[B].op.reduce_axis[0]
        ko, _ = s[B].split(k, nparts=nthx)
        s[B].bind(ko, thread_x)
        xo, xi = s[B].split(s[B].op.axis[0], factor=nthy)
        s[B].bind(xi, thread_y)
        s[B].bind(xo, block_x)

        tvm.lower(s, [A, B], simple_mode=True)

        # validation
        func = tvm.build(s, [A, B], device, name="warp_reduction")
        a_np = np.random.uniform(size=(m, n)).astype(A.dtype)
        b_np = np.zeros((m,), dtype=A.dtype)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        b_np = np.max(a_np, axis=1)
        func(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3, atol=1e-3)

    check_target("cuda", m=32, n=256)
    check_target("cuda", m=10, n=20)
    check_target("rocm", m=32, n=256)
    check_target("rocm", m=10, n=20)
    # This is a bug in normal reduction.
    # check_target("cuda", m=10, n=37)


@tvm.testing.requires_gpu
def test_warp_reduction2():
    def fcombine(x, y):
        return x[0] + y[0], x[1] * y[1]

    def fidentity(t0, t1):
        return tvm.tir.const(0, t0), tvm.tir.const(1, t1)

    add_mul_reducer = te.comm_reducer(fcombine, fidentity, name="add_mul_reducer")

    # compute
    m = 16
    n = 256
    A0 = te.placeholder((m, n), name="A0", dtype="float32")
    A1 = te.placeholder((m, n), name="Al", dtype="float32")
    k = te.reduce_axis((0, n), "k")
    T0, T1 = te.compute((m,), lambda i: add_mul_reducer((A0[i, k], A1[i, k]), axis=k), name="T")

    nthdx, nthdy = 32, 2
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, nthdx), "threadIdx.x")
    thread_y = te.thread_axis((0, nthdy), "threadIdx.y")

    def check_target(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return

        # schedule
        s = te.create_schedule(T0.op)
        ko, _ = s[T0].split(k, nparts=nthdx)
        xo, xi = s[T0].split(s[T0].op.axis[0], factor=nthdy)
        s[T0].bind(ko, thread_x)
        s[T0].bind(xi, thread_y)
        s[T0].bind(xo, block_x)

        # validation
        dev = tvm.device(device, 0)
        a0_np = np.random.uniform(size=(m, n)).astype(A0.dtype)
        a1_np = np.random.uniform(size=(m, n)).astype(A1.dtype)
        t0_np = np.zeros((m,), dtype=A0.dtype)
        t1_np = np.zeros((m,), dtype=A1.dtype)
        a0 = tvm.nd.array(a0_np, dev)
        a1 = tvm.nd.array(a1_np, dev)
        t0 = tvm.nd.array(t0_np, dev)
        t1 = tvm.nd.array(t1_np, dev)
        func = tvm.build(s, [A0, A1, T0, T1], device, name="reduction")
        func(a0, a1, t0, t1)
        t0_np = np.sum(a0_np, axis=1)
        t1_np = np.product(a1_np, axis=1)
        tvm.testing.assert_allclose(t0.numpy(), t0_np, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(t1.numpy(), t1_np, rtol=1e-3, atol=1e-3)

    check_target("cuda")
    check_target("rocm")


@tvm.testing.requires_gpu
def test_reduce_storage_reuse():
    target = tvm.target.Target("cuda")

    def run_passes(sch, args):
        func = schedule_to_primfunc(sch, args)
        mod = tvm.IRModule.from_expr(func)
        mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod)
        return tvm.transform.Sequential(
            [
                tvm.tir.transform.StorageFlatten(64),
                tvm.tir.transform.Simplify(),
                tvm.tir.transform.StorageRewrite(),
                tvm.tir.transform.LowerThreadAllreduce(),
            ]
        )(mod)

    dev = tvm.device(target.kind.name, 0)
    shape = (16, 16)

    A = te.placeholder(shape, dtype="float32", name="A")
    B = topi.nn.softmax(A, axis=1) + 1.0

    with tvm.target.Target(target):
        s = topi.cuda.schedule_softmax(B)

    mod = run_passes(s, [A, B])

    # Due to the storage rewrite pass, the reduction output storage reduce_temp0 can be reused as
    # the storage of the next compute.

    # Example:
    # ...
    # tir.tvm_thread_allreduce((uint32)1, normal_reduce_temp0[0], 1, reduce_temp0, threadIdx.x)
    # if ((threadIdx.x < 16)) {
    #   reduce_temp0[0] = (T_softmax_exp[threadIdx.x]/reduce_temp0[0])
    # }
    # ...

    # The LowerThreadAllreduce pass should remap reduce_temp0 on the left hand side of the store
    # above, as well as the load on the right hand side.

    # Expected output:
    # ...
    # red_buf0[0] = tir.tvm_warp_shuffle(mask[0], red_buf0[0], 0, 32, 32)
    # if ((threadIdx.x < 16)) {
    #   red_buf0[0] = (T_softmax_exp[threadIdx.x]/red_buf0[0])
    # }
    # ...

    def check_store_dst_remapped(op):
        if isinstance(op, tvm.tir.Store):
            assert op.buffer_var.name != "reduce_temp0"

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, check_store_dst_remapped)

    inp = np.random.uniform(size=shape).astype("float32")
    ref = tvm.topi.testing.softmax_python(inp) + 1.0

    f = tvm.build(s, [A, B], target)
    a = tvm.nd.array(inp, dev)
    b = tvm.nd.array(np.zeros(shape, dtype=B.dtype), dev)
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), ref, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__pfile__])
