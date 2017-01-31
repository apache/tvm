import tvm
import numpy as np

def test_sum():
    # graph
    n = tvm.Var('n')
    m = tvm.Var('m')
    A = tvm.placeholder((n, m), name='A')
    k = tvm.IterVar((0, m))
    B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name='B')
    # schedule
    s = tvm.Schedule(B.op)
    # create iter var and assign them tags.
    num_thread = 1
    block_x = tvm.IterVar(thread_tag="blockIdx.x")
    thread_x = tvm.IterVar((0, num_thread), thread_tag="threadIdx.x")
    _, x = s[B].split(B.op.axis[0], factor=num_thread, outer=block_x)
    _, x = s[B].split(x, outer=thread_x)

    tvm.init_opencl()
    codes = []
    fsum = tvm.build(s,
                     args=[A, B],
                     target="opencl", name="myadd",
                     record_codes=codes)
    for c in codes:
        print(c)
    num_device = 1
    for i in range(num_device):
        ctx = tvm.opencl(i)
        if not ctx.enabled:
            continue
        # launch the kernel.
        n = 1028
        m = 129
        #a = tvm.nd.array(np.zeros((n, m)).astype(A.dtype), ctx)
        a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
        b  = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        fsum(a, b)
        np.testing.assert_allclose(
            b.asnumpy(), np.sum(a.asnumpy(), axis=1), rtol=1e-4)


if __name__ == "__main__":
    test_sum()
