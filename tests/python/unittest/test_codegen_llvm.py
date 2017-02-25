import tvm
import numpy as np

def test_llvm_add_pipeline():
    n = tvm.Var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.Schedule(C.op)
    s[C].parallel(C.op.axis[0])

    def check_llvm():
        if not tvm.codegen.enabled("llvm"):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, B, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = 10270 * 2460
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        for i in range(1000):
            f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
    check_llvm()


if __name__ == "__main__":
    test_llvm_add_pipeline()
