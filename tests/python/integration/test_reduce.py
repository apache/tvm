import tvm
import numpy as np

def test_reduce_prims():
    def test_prim(reducer, np_reducer):
        # graph
        n = tvm.var('n')
        m = tvm.var('m')
        A = tvm.placeholder((n, m), name='A')
        k = tvm.reduce_axis((0, m))
        B = tvm.compute((n,), lambda i: reducer(A[i, k], axis=k, where=(i>1)), name='B')
        # schedule
        s = tvm.create_schedule(B.op)
        # create iter var and assign them tags.
        num_thread = 1
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
        s[B].bind(xi, tvm.thread_axis("threadIdx.x"))

        # one line to build the function.
        def check_device(device, host="stackvm"):
            if not tvm.module.enabled(host):
                return
            if not tvm.module.enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            ctx = tvm.context(device, 0)
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
            np.testing.assert_allclose(npy, res, rtol=1e-4)

        check_device("metal")
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
        np.testing.assert_allclose(
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
        if not tvm.module.enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        ctx = tvm.context(device, 0)
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
        np.testing.assert_allclose(
            b.asnumpy(), res, rtol=1e-4)

    check_target("cuda")
    check_target("metal")
    check_target("opencl")

if __name__ == "__main__":
    test_rfactor_threads()
    test_rfactor()
    test_reduce_prims()
