import tvm
import numpy as np
from tvm.contrib import mps

def test_matmul_add():
    if not tvm.module.enabled("metal"):
        print("skip because %s is not enabled..." % target)
        return
    n = 1024
    l = 128
    m = 256
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = mps.matmul(A, B)
    s1 = tvm.create_schedule(C.op)

    def verify(A, B, C, target="llvm"):
        if not tvm.get_global_func("tvm.contrib.mps.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.metal(0)
        f = tvm.build(s1, [A, B, C], "metal")
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)
    verify(A, B, C, s1)


def test_conv2d():
    if not tvm.module.enabled("metal"):
        print("skip because %s is not enabled..." % target)
        return
    n = 1
    h = 13
    w = 13
    ci = 32
    co = 128
    kh = 3
    kw = 3
    A = tvm.placeholder((n, h, w, ci), name="x")
    B = tvm.placeholder((co, kh, kw, ci), name="w")
    C = mps.conv2d(A, B)
    s1 = tvm.create_schedule(C.op)

    def verify(A, B, C, target="llvm"):
        if not tvm.get_global_func("tvm.contrib.mps.conv2d", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.metal(0)
        f = tvm.build(s1, [A, B, C], "metal")
        a = tvm.nd.array(np.random.uniform(size=(n, h, w, ci)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(co, kh, kw, ci)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, h, w, co), dtype=C.dtype), ctx)
        f(a, b, c)
        # print(c.asnumpy())
        #np.testing.assert_allclose(
        #    c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)
    verify(A, B, C, s1)


if __name__ == "__main__":
    test_conv2d()
