import tvm
import numpy as np
from tvm.contrib import nnpack

def test_fully_connected_output():
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    C = nnpack.fully_connected_inference(A, B)
    C = nnpack.fully_connected_output(A, B)
    D = tvm.compute(C.shape, lambda i, j: C[i,j] + bias, name="D")
    s = tvm.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_output", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        np.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T) + bb, rtol=1e-5)
    verify()


def test_fully_connected_inference():
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((l, ), name='A')
    B = tvm.placeholder((m, l), name='B')
    C = nnpack.fully_connected_inference(A, B)
    D = tvm.compute(C.shape, lambda i: C[i] + bias, name="D")
    s = tvm.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.nnpack.fully_connected_inference", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=(l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((m, ), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        np.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T) + bb, rtol=1e-5)
    verify()


if __name__ == "__main__":
    test_fully_connected_inference()
    test_fully_connected_output()
