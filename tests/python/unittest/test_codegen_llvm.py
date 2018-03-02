import tvm
import numpy as np
import ctypes

def test_llvm_intrin():
    ib = tvm.ir_builder.create()
    n = tvm.convert(4)
    A = ib.pointer("float32", name="A")
    args = [
        tvm.call_pure_intrin("handle", "tvm_address_of", A[0]),
        0, 3, 1
    ]
    ib.emit(tvm.make.Evaluate(
        tvm.make.Call(
            "int32", "prefetch", args, tvm.expr.Call.Intrinsic, None, 0)))
    body = ib.get()
    func = tvm.ir_pass.MakeAPI(body, "prefetch", [A], 0, True)
    fcode = tvm.build(func, None, "llvm")

def test_llvm_add_pipeline():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    AA = tvm.compute((n,), lambda *i: A(*i), name='A')
    BB = tvm.compute((n,), lambda *i: B(*i), name='B')
    T = tvm.compute(A.shape, lambda *i: AA(*i) + BB(*i), name='T')
    C = tvm.compute(A.shape, lambda *i: T(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
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
        f = tvm.build(s, [A, B, C], "llvm", binds=binds)
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


def test_llvm_persist_parallel():
    n = 128
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='B')
    C = tvm.compute(A.shape, lambda *i: tvm.sqrt(B(*i)) * 2 + 2, name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=8)
    xo1, xo2 = s[C].split(xo, nparts=1)
    s[B].compute_at(s[C], xo1)
    s[B].parallel(s[B].op.axis[0])
    s[B].pragma(s[B].op.axis[0], "parallel_barrier_when_finish")
    s[C].parallel(xi)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xi, "parallel_stride_pattern")

    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        # BUILD and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        np.testing.assert_allclose(c.asnumpy(),
                                   np.sqrt(a.asnumpy() + 1) * 2 + 2,
                                   rtol=1e-5)

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


def test_llvm_vadd_pipeline():
    def check_llvm(n, lanes):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n,), name='A', dtype="float32x%d" % lanes)
        B = tvm.compute((n,), lambda i: A[i], name='B')
        C = tvm.compute((n,), lambda i: B[i] + tvm.const(1, A.dtype), name='C')
        s = tvm.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], nparts=2)
        _, xi = s[C].split(xi, factor=2)
        s[C].parallel(xo)
        s[C].vectorize(xi)
        s[B].compute_at(s[C], xo)
        xo, xi = s[B].split(B.op.axis[0], factor=2)
        s[B].vectorize(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.empty((n,), A.dtype).copyfrom(
            np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), C.dtype, ctx)
        f(a, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + 1)
    check_llvm(64, 2)
    check_llvm(512, 2)


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



def test_llvm_select():
    def check_llvm(n, offset):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n, ), name='A')
        C = tvm.compute((n,), lambda i: tvm.select(i >= offset, A[i], 0.0), name='C')
        s = tvm.create_schedule(C.op)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
        c = tvm.nd.empty((n,), A.dtype, ctx)
        f(a, c)
        c_np = a.asnumpy()
        c_np[:offset] = 0
        np.testing.assert_allclose(c.asnumpy(), c_np)
    check_llvm(64, 8)


def test_llvm_bool():
    def check_llvm(n):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n, ), name='A', dtype="int32")
        C = tvm.compute((n,), lambda i: A[i].equal(1).astype("float"), name='C')
        s = tvm.create_schedule(C.op)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
        c = tvm.nd.empty((n,), C.dtype, ctx)
        f(a, c)
        c_np = a.asnumpy() == 1
        np.testing.assert_allclose(c.asnumpy(), c_np)
    check_llvm(64)


def test_rank_zero():
    def check_llvm(n):
        if not tvm.module.enabled("llvm"):
            return
        A = tvm.placeholder((n, ), name='A')
        scale = tvm.placeholder((), name='scale')
        k = tvm.reduce_axis((0, n), name="k")
        C = tvm.compute((), lambda : tvm.sum(A[k] * scale, axis=k), name="C")
        D = tvm.compute((), lambda : C + 1)
        s = tvm.create_schedule(D.op)
        # build and invoke the kernel.
        f = tvm.build(s, [A, scale, D], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), ctx)
        sc = tvm.nd.array(
            np.random.randint(0, 2, size=()).astype(scale.dtype), ctx)
        d = tvm.nd.empty((), D.dtype, ctx)
        f(a, sc, d)
        d_np = np.sum(a.asnumpy()) * sc.asnumpy() + 1
        np.testing.assert_allclose(d.asnumpy(), d_np)
    check_llvm(64)


def test_alignment():
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda i: A[i] * 3, name='B')
    s = tvm.create_schedule(B.op)
    bx, tx = s[B].split(B.op.axis[0], factor=8)
    s[B].vectorize(tx)
    f = tvm.build(s, [A, B], "llvm")

    for l in f.get_source().split("\n"):
        if "align" in l and "4 x float" in l:
            assert "align 32" in l


if __name__ == "__main__":
    test_alignment()
    test_rank_zero()
    test_llvm_bool()
    test_llvm_persist_parallel()
    test_llvm_select()
    test_llvm_vadd_pipeline()
    test_llvm_add_pipeline()
    test_llvm_intrin()
    test_multiple_func()
    test_llvm_flip_pipeline()
    test_llvm_madd_pipeline()
    test_llvm_temp_space()
