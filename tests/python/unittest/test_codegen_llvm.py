import tvm
import numpy as np

def test_llvm_add_pipeline():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    T = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='T')
    C = tvm.compute(A.shape, lambda *i: T(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # Specifically allow offset to test codepath when offset is available
        Ab = tvm.decl_buffer(
            A.shape, A.dtype,
            elem_offset=tvm.var('Aoffset'),
            offset_factor=8,
            name='A')
        binds = {A : Ab}
        # BUILD and invoke the kernel.
        f = tvm.build(s, [Ab, B, C], "llvm", binds=binds)
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())

    with tvm.build_config(offset_factor=4):
        check_llvm()


def test_llvm_flip_pipeline():
    def check_llvm(nn, base):
        if not tvm.module.enabled("llvm"):
            return
        n = tvm.convert(nn)
        A = tvm.placeholder((n + base), name='A')
        C = tvm.compute((n,), lambda i: A(nn + base- i - 1), name='C')
        s = tvm.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], factor=4)
        s[C].parallel(xo)
        s[C].vectorize(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=(n + base)).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy()[::-1][:n])
    check_llvm(4, 0)
    check_llvm(128, 8)
    check_llvm(3, 0)
    check_llvm(128, 1)


def test_llvm_madd_pipeline():
    def check_llvm(nn, base, stride):
        if not tvm.module.enabled("llvm"):
            return
        n = tvm.convert(nn)
        A = tvm.placeholder((n + base, stride), name='A')
        C = tvm.compute((n, stride), lambda i, j: A(base + i, j) + 1, name='C')
        s = tvm.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], factor=4)
        s[C].parallel(xo)
        s[C].vectorize(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=(n + base, stride)).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, stride), dtype=C.dtype), ctx)
        f(a, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy()[base:] + 1)
    check_llvm(64, 0, 2)
    check_llvm(4, 0, 1)
    with tvm.build_config(restricted_func=False):
        check_llvm(4, 0, 3)

def test_llvm_temp_space():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda i: A(i) + 1, name='B')
    C = tvm.compute(A.shape, lambda i: B(i) + 1, name='C')
    s = tvm.create_schedule(C.op)

    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + 1 + 1)
    check_llvm()

def test_multiple_func():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    s[C].parallel(xo)
    s[C].vectorize(xi)
    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # build two functions
        f2 = tvm.lower(s, [A, B, C], name="fadd1")
        f1 = tvm.lower(s, [A, B, C], name="fadd2")
        m = tvm.build([f1, f2], "llvm")
        fadd1 = m['fadd1']
        fadd2 = m['fadd2']
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd1(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
        fadd2(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
    check_llvm()


if __name__ == "__main__":
    test_multiple_func()
    test_llvm_add_pipeline()
    test_llvm_flip_pipeline()
    test_llvm_madd_pipeline()
    test_llvm_temp_space()
