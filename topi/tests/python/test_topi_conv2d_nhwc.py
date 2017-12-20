"""Example code to do convolution."""
import os
import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple


def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((num_filter, kernel, kernel, in_channel), name='W')
    B = topi.nn.conv2d_nhwc(A, W, stride, padding)
    with tvm.target.create("llvm"):
        s1 = topi.generic.schedule_conv2d_nhwc([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)

    @memoize("topi.tests.test_topi_conv2d_nhwc.verify_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        print("foo")
        b_np = topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func1 = tvm.build(s1, [A, W, B], device)
        func1(a, w, b)
        print(tvm.lower(s1, [A, W, B], simple_mode=True))
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        timer_f = func1.time_evaluator(func1.entry_name, ctx, number=5)
        t = timer_f(a, w, b).mean
        print("NHWC Time: ", t)

    for device in ['llvm']:
        check_device(device)


def test_conv2d_nhwc():
    verify_conv2d_nhwc(12, 32, 256, 16, 3, 1, "SAME")
    # verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "SAME")
    # verify_conv2d_nhwc(4, 128, 16, 128, 5, 2, "SAME")
    # verify_conv2d_nhwc(4, 128, 16, 256, 5, 2, "SAME")
    # verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "VALID")
    # verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "VALID")
    # verify_conv2d_nhwc(4, 128, 16, 128, 5, 2, "VALID")
    # verify_conv2d_nhwc(4, 128, 16, 256, 5, 2, "VALID")


if __name__ == "__main__":
    test_conv2d_nhwc()
