"""Test code for softmax"""
import os
import numpy as np
import tvm
import topi
import logging
import topi.testing
from topi.util import get_const_tuple

def verify_softmax(m, n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.cpp.nn.softmax(A, 1)
    # confirm lower works
    s = tvm.create_schedule([B.op])
    tvm.lower(s, [A, B], simple_mode=True)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = topi.testing.softmax_python(a_np)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.default_schedule(target, [B], False)
        else:
            s = topi.cpp.cuda.schedule_softmax(target, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        foo = tvm.build(s, [A, B], device, name="softmax")
        foo(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm']:
        check_device(device)

def test_softmax():
    verify_softmax(32, 10)
    verify_softmax(3, 4)


def verify_log_softmax(m, n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.cpp.nn.log_softmax(A)
    # confirm lower works
    s = tvm.create_schedule([B.op])
    tvm.lower(s, [A, B], simple_mode=True)
    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = topi.testing.log_softmax_python(a_np)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.default_schedule(target, [B], False)
        else:
            s = topi.cpp.cuda.schedule_softmax(target, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        foo = tvm.build(s, [A, B], device, name="log_softmax")
        foo(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ["cuda", "opencl", "metal", "rocm"]:
        check_device(device)


def test_log_softmax():
    verify_log_softmax(32, 10)
    verify_log_softmax(3, 4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_softmax()
    test_log_softmax()
