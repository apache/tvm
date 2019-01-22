import tvm
import numpy as np
from tvm.contrib import fbgemm


def test_matmul_fp16():
    n = 1024
    k = 128
    m = 235
    A = tvm.placeholder((m, k), name='A', dtype="int")
    B = tvm.placeholder((k, n), name='B', dtype="int")
    C = fbgemm.matmul_fp16(A, B)
    D = tvm.compute((m, n), lambda i, j: C[i][j], name="D")
    s = tvm.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.fbgemm.matmul_fp16", True):
            print("skip because extern function is not available")
            return

        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D], target)
        a = tvm.nd.array(np.random.uniform(0, 4, size=(m, k)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(0, 4, size=(k, n)).astype(B.dtype), ctx)
	d = tvm.nd.array(np.zeros((m, n), dtype=D.dtype), ctx)
        f(a, b, d)
        tvm.testing.assert_allclose(
            d.asnumpy(), np.matmul(a.asnumpy(), b.asnumpy()), rtol=1e-5)
    verify()


if __name__ == "__main__":
    # import nose
    # nose.runmodule()
    test_matmul_fp16()
