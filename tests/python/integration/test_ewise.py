import tvm
import numpy as np
import time

def test_exp():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: tvm.exp(A(*i)), name='B')
    s = tvm.create_schedule(B.op)
    # create iter var and assign them tags.
    num_thread = 8
    bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.module.enabled(host):
            return
        if not tvm.module.enabled(device):
            return
        fexp = tvm.build(s, [A, B],
                         device, host,
                         name="myexp")
        ctx = tvm.context(device, 0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        fexp(a, b)
        np.testing.assert_allclose(
            b.asnumpy(), np.exp(a.asnumpy()), rtol=1e-5)

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
    if not tvm.module.enabled("llvm"):
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

from tvm.contrib import nvcc_compiler

@tvm.register_func
def tvm_callback_cuda_compile(code):
    print(code)
    ptx =  nvcc_compiler.compile_source(code, target="ptx", options=["-arch=sm_52"])
    return ptx

def test_add():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    bias = tvm.var("bias", dtype="float32")
    scale = tvm.var("scale", dtype="float32")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i) * scale + bias, name='C')
    # schedule
    s = tvm.create_schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 32
    bx, x = s[C].split(C.op.axis[0], factor=num_thread*4)
    tx, x = s[C].split(x, nparts=num_thread)
    _, x = s[C].split(x, factor=4)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[C].vectorize(x)

    # one line to build the function.
    def check_device(device):
        if not tvm.module.enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        fadd = tvm.build(s, [A, B, C, bias, scale],
                         device,
                         name="myadd")
        ctx = tvm.context(device, 0)
        print(fadd.imported_modules[0].get_source())
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        vbias = np.random.uniform()
        vscale = np.random.uniform()
        fadd(a, b, c, vbias, vscale)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy() * vscale + vbias, rtol=1e-6)

    check_device("opencl")
    check_device("metal")
    check_device("cuda")


if __name__ == "__main__":
    test_log_llvm()
    test_exp()
    test_add()
