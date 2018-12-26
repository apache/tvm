"""Test code for softmax"""
import os
import numpy as np
import tvm
import topi
import topi.testing
import logging
from topi.util import get_const_tuple

from common import get_all_backend

def check_device(A, B, a_np, b_np, device, name):
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    print("Running on target: %s" % device)
    with tvm.target.create(device):
        s = topi.generic.schedule_softmax(B)

    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    f = tvm.build(s, [A, B], device, name="softmax")
    f(a, b)
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def verify_softmax(m, n, dtype="float32"):
    A = tvm.placeholder((m, n), dtype=dtype, name='A')
    B = topi.nn.softmax(A)
    # confirm lower works
    s = tvm.create_schedule([B.op])
    tvm.lower(s, [A, B], simple_mode=True)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = topi.testing.softmax_python(a_np)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx']:
        check_device(A, B, a_np, b_np, device, "softmax")

def verify_softmax_4d(shape, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = topi.nn.softmax(A, axis=1)

    _, c, h, w = shape
    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = topi.testing.softmax_python(a_np.transpose(0, 2, 3, 1).reshape(h*w, c))
    b_np = b_np.reshape(1, h, w, c).transpose(0, 3, 1, 2)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx']:
        check_device(A, B, a_np, b_np, device, "softmax")

def test_softmax():
    verify_softmax(32, 10)
    verify_softmax(3, 4)
    verify_softmax(32, 10, "float64")
    verify_softmax_4d((1, 16, 256, 256))

def verify_log_softmax(m, n, dtype="float32"):
    A = tvm.placeholder((m, n), dtype=dtype, name='A')
    B = topi.nn.log_softmax(A)
    # confirm lower works
    s = tvm.create_schedule([B.op])
    tvm.lower(s, [A, B], simple_mode=True)
    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = topi.testing.log_softmax_python(a_np)

    for device in get_all_backend():
        check_device(A, B, a_np, b_np, device, "log_softmax")


def test_log_softmax():
    verify_log_softmax(32, 10)
    verify_log_softmax(3, 4)
    verify_log_softmax(32, 10, "float64")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_softmax()
    test_log_softmax()
