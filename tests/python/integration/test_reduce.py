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
    block_x = tvm.thread_axis(None, "blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    _, x = s[B].split(B.op.axis[0], factor=num_thread, outer=block_x)
    _, x = s[B].split(x, outer=thread_x)

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
    B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k, where=(i>1)), name='B')
    kf = tvm.reduce_axis((0, 4))
    # schedule
    s = tvm.Schedule(B.op)
    _, ki = s[B].split(k, outer=kf)
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

if __name__ == "__main__":
    test_rfactor()
    test_sum()
