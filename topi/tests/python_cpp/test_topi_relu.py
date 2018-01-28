"""Test code for relu activation"""
import os
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def verify_relu(m, n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.cpp.nn.relu(A)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0)

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
        foo = tvm.build(s, [A, B], device, name="relu")
        foo(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm']:
        check_device(device)


def verify_leaky_relu(m, alpha):
    A = tvm.placeholder((m,), name='A')
    B = topi.cpp.nn.leaky_relu(A, alpha)
    device = "llvm"
    target = topi.cpp.TEST_create_target(device)
    s = topi.cpp.generic.schedule_injective(target, [B])

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0) + a_np * (a_np < 0) * alpha
    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    foo = tvm.build(s, [A, B], device, name="leaky_relu")
    foo(a, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)


def test_relu():
    verify_relu(10, 128)

def test_leaky_relu():
    verify_leaky_relu(100, 0.1)


if __name__ == "__main__":
    test_relu()
    test_leaky_relu()
