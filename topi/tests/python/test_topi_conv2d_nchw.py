"""Example code to do convolution."""
import os
import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d.verify_con2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            B = topi.nn.conv2d(A, W, stride, padding, layout='NCHW')
            C = topi.nn.relu(B)
            s1 = topi.generic.schedule_conv2d_nchw([B])
            s2 = topi.generic.schedule_conv2d_nchw([C])
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device != "cuda")):
            func1 = tvm.build(s1, [A, W, B], device, name="conv2d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding))
            func2 = tvm.build(s2, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding))
            func1(a, w, b)
            func2(a, w, c)
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)


def test_conv2d_nchw():
    # ResNet18 worklaods
    verify_conv2d_nchw(1, 3, 224, 64, 7, 2, 3)
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nchw(1, 64, 56, 64, 1, 1, 0)
    verify_conv2d_nchw(1, 64, 56, 128, 3, 2, 1)
    verify_conv2d_nchw(1, 64, 56, 128, 1, 2, 0)
    verify_conv2d_nchw(1, 128, 28, 128, 3, 1, 1)
    verify_conv2d_nchw(1, 128, 28, 256, 3, 2, 1)
    verify_conv2d_nchw(1, 128, 28, 256, 1, 2, 0)
    verify_conv2d_nchw(1, 256, 14, 256, 3, 1, 1)
    verify_conv2d_nchw(1, 256, 14, 512, 3, 2, 1)
    verify_conv2d_nchw(1, 256, 14, 512, 1, 2, 0)
    verify_conv2d_nchw(1, 512, 7, 512, 3, 1, 1)
    # Vgg16 workloads
    verify_conv2d_nchw(1, 128, 122, 128, 3, 1, 1)
    # Super resolution workloads
    verify_conv2d_nchw(1, 1, 224, 64, 5, 1, 2)
    verify_conv2d_nchw(1, 64, 224, 64, 3, 1, 1)
    verify_conv2d_nchw(1, 64, 224, 32, 3, 1, 1)
    verify_conv2d_nchw(1, 32, 224, 9, 3, 1, 1)

if __name__ == "__main__":
    test_conv2d_nchw()
