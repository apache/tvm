"""Test code for relu activation"""
import os
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

from common import get_all_backend

def _verify_elemwise(shape, np_func, fn_name, *fn_args):
    A = tvm.placeholder(shape, name='A')
    B = getattr(topi.nn, fn_name)(A, *fn_args)

    a_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = np_func(a_np)

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
        foo = tvm.build(s, [A, B], device, name=fn_name)
        foo(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)


def verify_relu(*shape):
    return _verify_elemwise(shape, lambda x: x * (x > 0), 'relu')


def verify_glu(shape, axis):
    def _glu(x):
        a, b = np.split(x, 2, axis=axis)
        return a * 1/(1 + np.exp(-b))
    return _verify_elemwise(shape, _glu, 'glu', axis)


def verify_leaky_relu(m, alpha):
    return _verify_elemwise(
        (m,), lambda x: x * (x > 0) + x * (x < 0) * alpha, 'leaky_relu', alpha)


def verify_prelu(x, w, axis, weight_reshape):
    X = tvm.placeholder((x), name='X')
    W = tvm.placeholder((w), name='W')
    x_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(X.shape)).astype(X.dtype)
    w_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(W.shape)).astype(W.dtype)

    def _prelu_numpy(x, W):
        return (x < 0) * (x *W.reshape(weight_reshape)) + (x>=0) * x

    B = topi.nn.prelu(X, W, axis)
    s = tvm.create_schedule([B.op])

    ctx = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, ctx)
    w_tvm = tvm.nd.array(w_np, ctx)

    b = tvm.nd.array(np.zeros(get_const_tuple(X.shape), dtype=B.dtype), ctx)
    foo = tvm.build(s, [X, W, B], "llvm", name="prelu")
    foo(x_tvm, w_tvm, b)
    out_np = _prelu_numpy(x_np, w_np)
    tvm.testing.assert_allclose(b.asnumpy(), out_np, rtol=1e-5)


def test_relu():
    verify_relu(10, 128)

def test_schedule_big_array():
    verify_relu(1024 * 100 , 512)


def test_leaky_relu():
    verify_leaky_relu(100, 0.1)

def test_prelu():
    verify_prelu((1, 3, 2, 2), (3,), 1, (3, 1, 1))
    verify_prelu((1, 3, 2, 2), (2,), 2, (2, 1))

def test_glu():
    verify_glu((10, 24, 42), axis=1)
    verify_glu((10, 24, 42), axis=-1)

if __name__ == "__main__":
    test_schedule_big_array()
    test_relu()
    test_leaky_relu()
    test_prelu()
    test_glu()
