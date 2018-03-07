import tvm
import numpy as np
from tvm.contrib import mps

def test_matmul():
    if not tvm.module.enabled("metal"):
        print("skip because %s is not enabled..." % "metal")
        return
    n = 1024
    l = 128
    m = 256
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = mps.matmul(A, B)
    D = tvm.compute(
        C.shape,
        lambda *i: C(*i) + 1.
    )
    s = tvm.create_schedule(D.op)
    yo, xo = D.op.axis
    block_y = tvm.thread_axis("blockIdx.y")
    block_x = tvm.thread_axis("blockIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    thread_x = tvm.thread_axis("threadIdx.x")
    by, ty = s[D].split(yo, factor=16)
    bx, tx = s[D].split(xo, factor=16)
    s[D].bind(by, block_y)
    s[D].bind(bx, block_x)
    s[D].bind(ty, thread_y)
    s[D].bind(tx, thread_x)



    def verify(A, B, D, s, target="metal"):
        if not tvm.get_global_func("tvm.contrib.mps.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.metal(0)
        f = tvm.build(s, [A, B, D], "metal")
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + 1, rtol=1e-5)
    verify(A, B, D, s)

def test_conv2d():
    if not tvm.module.enabled("metal"):
        print("skip because %s is not enabled..." % "metal")
        return
    n = 1
    h = 14
    w = 14
    ci = 2
    co = 4
    kh = 3
    kw = 3
    stride = 2
    A = tvm.placeholder((n, h, w, ci), name="x")
    B = tvm.placeholder((co, kh, kw, ci), name="w")
    C = mps.conv2d(A, B, 'SAME', 2)
    s1 = tvm.create_schedule(C.op)

    def verify(A, B, C, target="llvm"):
        if not tvm.get_global_func("tvm.contrib.mps.conv2d", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.metal(0)
        f = tvm.build(s1, [A, B, C], "metal")
        a = tvm.nd.array(np.random.uniform(size=(n, h, w, ci)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(co, kh, kw, ci)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, h // stride, w // stride, co), dtype=C.dtype), ctx)
        f(a, b, c)
        # print(c.asnumpy())
        # print(c.shape)
        
    verify(A, B, C, s1)


if __name__ == "__main__":
    #test_matmul()
    test_conv2d()

