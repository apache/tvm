import tvm
from tvm.contrib import nvcc
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
        ctx = tvm.context(device, 0)
        if not ctx.exist:
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

    check_device("opencl -device=intel_graphics")
    check_device("cuda", "llvm")
    check_device("vulkan")


def test_multiple_cache_write():
    # graph
    n = tvm.convert(1024)
    A0 = tvm.placeholder((n,), name='A0', dtype = "float32")
    A1 = tvm.placeholder((n,), name='A1', dtype = "float32")
    B0, B1 = tvm.compute((n,),
            lambda *i: (A0(*i) + A1(*i), A0(*i) * A1(*i)),
            name='B')
    C = tvm.compute((n,), lambda *i: B0(*i) + B1(*i),
            name='C')
    s = tvm.create_schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 8
    B0_cache, B1_cache = s.cache_write([B0, B1], "local")
    bx, tx = s[C].split(C.op.axis[0], factor=num_thread)
    s[B0].compute_at(s[C], bx)
    s[B0_cache].compute_at(s[C], bx)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.module.enabled(host):
            return
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            return
        func = tvm.build(s, [A0, A1, C],
                         device, host,
                         name="multiple_cache_write")
        ctx = tvm.context(device, 0)
        # launch the kernel.
        n = 1024
        a0 = tvm.nd.array(np.random.uniform(size=n).astype(A0.dtype), ctx)
        a1 = tvm.nd.array(np.random.uniform(size=n).astype(A1.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        func(a0, a1, c)
        np.testing.assert_allclose(
            c.asnumpy(), a0.asnumpy() + a1.asnumpy() + (a0.asnumpy() * a1.asnumpy()),
            rtol=1e-5)

    check_device("cuda", "llvm")
    check_device("vulkan")
    check_device("opencl")

def test_log_pow_llvm():
    # graph
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: tvm.power(tvm.log(A(*i)), 2.0), name='B')
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
    repeat = 10
    ftimer = flog.time_evaluator(flog.entry_name, ctx, number=1, repeat=repeat)
    res = ftimer(a, b)
    assert(len(res.results) == repeat)
    np.testing.assert_allclose(
        b.asnumpy(), np.power(np.log(a.asnumpy()), 2.0), rtol=1e-5)


def test_popcount():
    def run(dtype):
        # graph
        n = tvm.convert(1024)
        A = tvm.placeholder((n,), name='A', dtype=dtype)
        B = tvm.compute(A.shape, lambda *i: tvm.popcount(A(*i)), name='B')
        s = tvm.create_schedule(B.op)
        # simple schedule
        num_thread = 8
        bx, tx = s[B].split(B.op.axis[0], factor=num_thread)

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("skip because %s is not enabled.." % device)
                return
            target = tvm.target.create(device)
            if "cpu" not in target.keys:
                s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
                s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
            func = tvm.build(s, [A, B], device)
            # launch the kernel.
            n = 1024
            a = tvm.nd.array(np.random.randint(low=0, high=1000, size=n, dtype=A.dtype), ctx)
            b = tvm.nd.array(np.zeros(shape=n, dtype=B.dtype), ctx)
            func(a, b)
            np.testing.assert_allclose(
                b.asnumpy(), list(map(lambda x: bin(x).count('1'), a.asnumpy())), rtol=1e-5)

        check_device("llvm")
        check_device("cuda")
        check_device("opencl")
        if dtype == "uint32":
            check_device("metal")
            check_device("vulkan")
    run('uint32')
    run('uint64')


def test_add():
    def run(dtype):
        # graph
        n = tvm.var('n')
        A = tvm.placeholder((n,), name='A', dtype=dtype)
        B = tvm.placeholder((n,), name='B', dtype=dtype)
        bias = tvm.var("bias", dtype=dtype)
        scale = tvm.var("scale", dtype=dtype)
        C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        # schedule
        s = tvm.create_schedule(C.op)
        # create iter var and assign them tags.
        num_thread = 16
        bx, x = s[C].split(C.op.axis[0], factor=num_thread*4)
        tx, x = s[C].split(x, nparts=num_thread)
        _, x = s[C].split(x, factor=4)
        s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[C].vectorize(x)

        # one line to build the function.
        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("skip because %s is not enabled.." % device)
                return
            fadd = tvm.build(s, [A, B, C],
                             device,
                             name="myadd")

            # launch the kernel.
            n = 1024
            a = tvm.nd.array((np.random.uniform(size=n) * 256).astype(A.dtype), ctx)
            b = tvm.nd.array((np.random.uniform(size=n) * 256).astype(B.dtype), ctx)
            c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
            ftimer = fadd.time_evaluator(fadd.entry_name, ctx, number=1)
            tcost = ftimer(a, b, c).mean
            np.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy(), rtol=1e-6)

        check_device("opencl")
        check_device("cuda")
        if dtype == "float32":
            check_device("metal")
            check_device("vulkan")

    run("float32")
    run("int32")
    run("int64")
    run("uint64")


def try_warp_memory():
    """skip this in default test because it require higher arch"""
    m = 128
    A = tvm.placeholder((m,), name='A')
    B = tvm.compute((m,), lambda i: A[i] + 3, name='B')
    warp_size = 32
    s = tvm.create_schedule(B.op)
    AA = s.cache_read(A, "warp", [B])
    xo, xi = s[B].split(B.op.axis[0], warp_size * 2)
    xi0, xi1 = s[B].split(xi, factor=warp_size)
    tx = tvm.thread_axis("threadIdx.x")
    s[B].bind(xi1, tx)
    s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[AA].compute_at(s[B], xo)
    xo, xi = s[AA].split(s[AA].op.axis[0], warp_size)
    s[AA].bind(xi, tx)

    @tvm.register_func
    def tvm_callback_cuda_compile(code):
        ptx =  nvcc.compile_cuda(code, target="ptx")
        return ptx

    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("skip because %s is not enabled.." % device)
            return
        f = tvm.build(s, [A, B], device)
        a = tvm.nd.array((np.random.uniform(size=m) * 256).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(m, dtype=B.dtype), ctx)
        f(a, b)
        np.testing.assert_allclose(
            b.asnumpy(), a.asnumpy() + 3, rtol=1e-6)
    check_device("cuda")


if __name__ == "__main__":
    test_exp()
    try_warp_memory()
    test_multiple_cache_write()
    test_add()
    test_log_pow_llvm()
    test_popcount()
