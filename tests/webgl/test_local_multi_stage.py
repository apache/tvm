import tvm
import numpy as np

def test_local_multi_stage():
    if not tvm.module.enabled("opengl"):
        return
    if not tvm.module.enabled("llvm"):
        return

    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype="int32")
    B = tvm.compute((n,), lambda i: A[i] + 1, name="B")
    C = tvm.compute((n,), lambda i: B[i] * 2, name="C")

    s = tvm.create_schedule(C.op)
    s[B].opengl()
    s[C].opengl()

    f = tvm.build(s, [A, C], "opengl", name="multi_stage")

    ctx = tvm.opengl(0)
    n = 10
    a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), ctx)
    f(a, c)

    np.testing.assert_allclose(c.asnumpy(), (a.asnumpy() + 1) * 2)

if __name__ == "__main__":
    test_local_multi_stage()
