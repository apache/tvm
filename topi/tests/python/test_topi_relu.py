"""Test code for relu activation"""
import os
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def verify_relu(m, n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.nn.relu(A)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_elemwise(B)

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        foo = tvm.build(s, [A, B], device, name="relu")
        foo(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)


def verify_leaky_relu(m, alpha):
    A = tvm.placeholder((m,), name='A')
    B = topi.nn.leaky_relu(A, alpha)
    s = tvm.create_schedule([B.op])

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0) + a_np * (a_np < 0) * alpha
    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    foo = tvm.build(s, [A, B], "llvm", name="leaky_relu")
    foo(a, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)


def verify_prelu(x, w):
    X = tvm.placeholder((x), name='X')
    W = tvm.placeholder((w), name='W')
    x_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(X.shape)).astype(X.dtype)
    w_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(W.shape)).astype(W.dtype)

    def _prelu_numpy(x, W):
        return (x < 0) * (x *W.reshape(3, 1, 1)) + (x>=0) * x

    B = topi.nn.prelu(X, W)
    s = tvm.create_schedule([B.op])

    ctx = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, ctx)
    w_tvm = tvm.nd.array(w_np, ctx)

    b = tvm.nd.array(np.zeros(get_const_tuple(X.shape), dtype=B.dtype), ctx)
    foo = tvm.build(s, [X, W, B], "llvm", name="prelu")
    foo(x_tvm, w_tvm, b)
    out_np = _prelu_numpy(x_np, w_np)
    np.testing.assert_allclose(b.asnumpy(), out_np, rtol=1e-5)

def test_relu():
    verify_relu(10, 128)

def test_leaky_relu():
    verify_leaky_relu(100, 0.1)

def test_prelu():
    verify_prelu((1, 3, 2, 2), (3,))

if __name__ == "__main__":
    test_relu()
    test_leaky_relu()
    test_prelu()
