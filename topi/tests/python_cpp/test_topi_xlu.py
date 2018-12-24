"""Test code for relu activation"""
import os
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def _verify_elemwise(shape, dtype, np_func, fn_name, *fn_args):
    A = tvm.placeholder(shape, name='A', dtype=dtype)
    B = getattr(topi.cpp.nn, fn_name)(A, *fn_args)
    assert B.dtype == dtype

    a_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = np_func(a_np)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.schedule_injective(target, [B])
        else:
            s = topi.cpp.cuda.schedule_injective(target, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        foo = tvm.build(s, [A, B], device, name=fn_name)
        foo(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm']:
        check_device(device)


def verify_relu(*shape, dtype):
    return _verify_elemwise(shape, dtype, lambda x: x * (x > 0), 'relu')


def verify_glu(*shape, dtype, axis):
    def _glu(x):
        a, b = np.split(x, 2, axis=axis)
        return a * 1/(1 + np.exp(-b))
    return _verify_elemwise(shape, dtype, _glu, 'glu', axis)


def verify_leaky_relu(m, alpha, dtype='float32'):
    return _verify_elemwise(
        (m,), dtype, lambda x: x * (x > 0) + x * (x < 0) * alpha, 'leaky_relu', alpha)


def verify_prelu(x, w, axis, weight_reshape):
    X = tvm.placeholder((x), name='X')
    W = tvm.placeholder((w), name='W')
    x_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(X.shape)).astype(X.dtype)
    w_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(W.shape)).astype(W.dtype)
    def _prelu_numpy(x, W):
        return (x < 0) * (x *W.reshape(weight_reshape)) + (x>=0) * x

    out_np = _prelu_numpy(x_np, w_np)
    B = topi.cpp.nn.prelu(X, W, axis)
    device = "llvm"
    target = topi.cpp.TEST_create_target(device)
    s = topi.cpp.generic.schedule_injective(target, [B])

    ctx = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, ctx)
    w_tvm = tvm.nd.array(w_np, ctx)

    b = tvm.nd.array(np.zeros(get_const_tuple(X.shape), dtype=B.dtype), ctx)
    foo = tvm.build(s, [X, W, B], "llvm", name="prelu")
    foo(x_tvm, w_tvm, b)
    tvm.testing.assert_allclose(b.asnumpy(), out_np, rtol=1e-5)


def test_relu():
    for dtype in ['float32', 'float64', 'int32', 'int16', 'int8', 'int64']:
        verify_relu(10, 128, dtype=dtype)

def test_glu():
    for i, dtype in enumerate(['float32', 'float64']):
        verify_glu(10, 128, dtype=dtype, axis=0)
        verify_glu(10, 128, dtype=dtype, axis=-1)

def test_leaky_relu():
    verify_leaky_relu(100, 0.5)

def test_prelu():
    verify_prelu((1, 3, 2, 2), (3,), 1, (3, 1, 1))
    verify_prelu((1, 3, 2, 2), (2,), 2, (2, 1))

if __name__ == "__main__":
    test_relu()
    test_leaky_relu()
    test_prelu()
    test_glu()
