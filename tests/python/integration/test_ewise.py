import tvm
import numpy as np

def test_add():
    # graph
    n = tvm.Var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    # schedule
    s = tvm.Schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 256
    block_x = tvm.IterVar(thread_tag="blockIdx.x")
    thread_x = tvm.IterVar((0, num_thread), thread_tag="threadIdx.x")
    _, x = s[C].split(C.op.axis[0], factor=num_thread, outer=block_x)
    _, x = s[C].split(x, outer=thread_x)

    # one line to build the function.
    codes = []
    fadd = tvm.build(s,
                     args=[A, B, C],
                     target="cuda", name="myadd",
                     record_codes=codes)
    for c in codes:
        print(c)

    # call the function
    num_device = 1
    for i in range(num_device):
        ctx = tvm.gpu(i)
        if not ctx.enabled:
            continue
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())


if __name__ == "__main__":
    test_add()
