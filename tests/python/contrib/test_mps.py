import tvm
import numpy as np
from tvm.contrib import mps

def test_matmul_add():
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C1 = mps.matmul(A, B)
    C2 = mps.matmul(B, A, True, True)
    D1 = tvm.compute(C1.shape, lambda i, j: C1[i,j] + bias, name="D1")
    D2 = tvm.compute(C2.shape, lambda i, j: C2[i,j] + bias, name="D2")
    s1 = tvm.create_schedule(D1.op)
    s2 = tvm.create_schedule(D2.op)

    def verify(A, B, D, s, bias, target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.mps.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        np.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + bb, rtol=1e-5)
    verify(A, B, D1, s1, bias)
    verify(A, B, D2, s2, bias)


if __name__ == "__main__":
    test_matmul_add()
