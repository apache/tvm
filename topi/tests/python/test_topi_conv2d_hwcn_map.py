"""Example code to do convolution."""
import os
import numpy as np
import tvm
import topi
from topi.nn.util import get_const_tuple


def verify_conv2d_hwcn_map(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((in_height, in_width, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.conv2d_hwcn(A, W, stride, padding)
    C = topi.nn.relu(B)
    s1 = topi.cuda.schedule_conv2d_hwcn_map(B.op)
    s2 = topi.cuda.schedule_conv2d_hwcn_map(C.op)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    w_np = np.random.uniform(size=get_const_tuple(W.shape)).astype(W.dtype)
    b_np = topi.testing.conv2d_hwcn_python(a_np, w_np, stride, padding)
    c_np = np.maximum(b_np, 0)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        with tvm.build_config(auto_unroll_max_step=32,
                              auto_unroll_min_depth=0,
                              unroll_explicit=False):
            func1 = tvm.build(s1, [A, W, B], device)
            func2 = tvm.build(s2, [A, W, C], device)
            func1(a, w, b)
            func2(a, w, c)
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal']:
        check_device(device)


def test_conv2d_hwcn_map():
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_hwcn_map(4, 128, 16, 128, 5, 2, "SAME")
    verify_conv2d_hwcn_map(4, 128, 16, 256, 5, 2, "SAME")
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_hwcn_map(4, 128, 16, 128, 5, 2, "VALID")
    verify_conv2d_hwcn_map(4, 128, 16, 256, 5, 2, "VALID")


if __name__ == "__main__":
    test_conv2d_hwcn_map()
