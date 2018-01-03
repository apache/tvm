import tvm
import numpy as np
from tvm.contrib import rocblas

def test_matmul_add():
    n = 1024
    l = 128
    m = 235
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = rocblas.matmul(A, B)
    s = tvm.create_schedule(C.op)

    def verify(target="rocm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.rocblas.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.rocm(0)
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)
    verify()


if __name__ == "__main__":
    test_matmul_add()
