import tvm
import numpy as np
from tvm.contrib import fbgemm
from collections import namedtuple


QuantParams = namedtuple("QuantParams", "scale zero_point") 
def test_fc_int8():
    n = 32
    k = 16
    m = 8
    X = tvm.placeholder((m, k), name='X', dtype="int8")
    W = tvm.placeholder((n, k), name='W', dtype="int8")
    B = tvm.placeholder((n,), name='B', dtype="int")
    # quantization parameters will be got from Operator arguments
    X_qparams = QuantParams(scale=1.0, zero_point=0)
    W_qparams = QuantParams(scale=1.0, zero_point=0)
    Y_qparams = QuantParams(scale=1.0, zero_point=0)
    C = fbgemm.fully_connected_int8(X, X_qparams, W, W_qparams, B, Y_qparams)
    Y = tvm.compute((m, n), lambda i, j: C[i][j], name="Y") 
    s = tvm.create_schedule(Y.op)


    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.fbgemm.fully_connected_int8", True):
            print("skip because extern function is not available")
            return

        ctx = tvm.cpu(0)
        f = tvm.build(s, [X, W, B, Y], target)
        x = tvm.nd.array(np.random.uniform(2, 2, size=(m, k)).astype(X.dtype), ctx)
        w = tvm.nd.array(np.random.uniform(1, 1, size=(n, k)).astype(W.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(0, 0, size=(n,)).astype(B.dtype), ctx)
	b_ = tvm.nd.array(np.zeros((m, n), dtype=B.dtype), ctx)
	y = tvm.nd.array(np.zeros((m, n), dtype=Y.dtype), ctx)
        f(x, w, b, y)
        tvm.testing.assert_allclose(
           y.asnumpy(), np.matmul(x.asnumpy(), w.asnumpy().T) + b_.asnumpy(), rtol=1e-5)
    verify()

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
    test_matmul_fp16()
    test_fc_int8()
