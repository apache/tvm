import tvm
import numpy as np
from tvm.contrib import cublas

def test_matmul_add():
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = cublas.matmul(A, B)
    D = tvm.compute(C.shape, lambda i, j: C[i,j] + bias, name="D")
    import topi
    with tvm.target.create("cuda -libs=cublas"):
        s = topi.generic.schedule_extern(D)

    def verify(target="cuda"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cublas.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.gpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        np.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + bb, rtol=1e-5)
    verify()


if __name__ == "__main__":
    test_matmul_add()
