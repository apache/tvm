import tvm
import numpy as np

def test_sum():
    # graph
    n = tvm.Var('n')
    m = tvm.Var('m')
    A = tvm.placeholder((n, m), name='A')
    k = tvm.reduce_axis((0, m))
    B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k, where=(i>1)), name='B')
    # schedule
    s = tvm.Schedule(B.op)
    # create iter var and assign them tags.
    num_thread = 1
    xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[B].bind(xi, tvm.thread_axis("threadIdx.x"))

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.codegen.enabled(host):
            return
        if not tvm.codegen.enabled(device):
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        fsum = tvm.build(s,
                         args=[A, B],
                         target=device, target_host=host,
                         name="mysum")
        print(fsum.imported_modules[0].get_source())
        # launch the kernel.
        n = 1028
        m = 129
        a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
        b  = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        fsum(a, b)
        res = np.sum(a.asnumpy(), axis=1)
        res[:2] = 0
        np.testing.assert_allclose(
            b.asnumpy(), res, rtol=1e-4)

    if tvm.module.enabled("opencl"):
        tvm.module.init_opencl()

    check_device("cuda")
    check_device("opencl")


def test_rfactor():
    n = tvm.convert(1027)
    A = tvm.placeholder((n,), name='A')
    k = tvm.reduce_axis((0, n))
    B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')
    # schedule
    s = tvm.Schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf)
    s[BF].parallel(BF.op.axis[0])
    # one line to build the function.
    def check_target(target="llvm"):
        if not tvm.codegen.enabled(target):
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
    s = tvm.Schedule(B.op)
    ko, kf = s[B].split(k, factor=nthread)
    BF = s.rfactor(B, kf)
    bx, tx = s[B].split(s[B].op.axis[0], factor=nthread)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.y"))
    s[B].bind(s[B].op.reduce_axis[0], tvm.thread_axis("threadIdx.x"))
    s[BF].compute_at(s[B], tx)

    # one line to build the function.
    def check_target(device, host="stackvm"):
        if not tvm.codegen.enabled(device):
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        fapi = tvm.lower(s, args=[A, B])
        fapi2 = tvm.ir_pass.LowerThreadAllreduce(fapi, 32)
        fsum = tvm.build(fapi,
                         target=device,
                         name="mysum")
        print(fsum.imported_modules[0].get_source())
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

    if tvm.module.enabled("opencl"):
        tvm.module.init_opencl()
    check_target("cuda")
    check_target("opencl")

if __name__ == "__main__":
    test_rfactor()
    test_rfactor_threads()
    test_sum()
