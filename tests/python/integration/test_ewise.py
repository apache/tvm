import tvm
import numpy as np

def test_exp():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: tvm.exp(A(*i)), name='B')
    s = tvm.create_schedule(B.op)
    # create iter var and assign them tags.
    num_thread = 64
    bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.codegen.enabled(host):
            return
        if not tvm.codegen.enabled(device):
            return
        fexp = tvm.build(s, [A, B],
                         device, host,
                         name="myexp")
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        fexp(a, b)
        np.testing.assert_allclose(
            b.asnumpy(), np.exp(a.asnumpy()), rtol=1e-5)

    if tvm.module.enabled("opencl"):
        tvm.module.init_opencl()
    check_device("cuda", "llvm")
    check_device("opencl")


def test_log_llvm():
    # graph
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: tvm.log(A(*i)), name='B')
    s = tvm.create_schedule(B.op)
    # create iter var and assign them tags.
    bx, tx = s[B].split(B.op.axis[0], factor=32)
    # one line to build the function.
    if not tvm.codegen.enabled("llvm"):
        return

    flog = tvm.build(s, [A, B],
                     "llvm", name="mylog")
    ctx = tvm.cpu(0)
    # launch the kernel.
    n = 1028
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
    flog(a, b)
    np.testing.assert_allclose(
        b.asnumpy(), np.log(a.asnumpy()), rtol=1e-5)


def test_add():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    # schedule
    s = tvm.create_schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 256
    bx, x = s[C].split(C.op.axis[0], factor=num_thread*4)
    tx, x = s[C].split(x, nparts=num_thread)
    _, x = s[C].split(x, factor=4)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[C].vectorize(x)

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.codegen.enabled(host):
            print("skip because %s is not enabled.." % host)
            return
        if not tvm.codegen.enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        fadd = tvm.build(s, [A, B, C],
                         device, host,
                         name="myadd")
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())

    if tvm.module.enabled("opencl"):
        tvm.module.init_opencl()
    check_device("cuda", "llvm")
    check_device("opencl")


if __name__ == "__main__":
    test_log_llvm()
    test_exp()
    test_add()
