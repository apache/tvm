"""Example code to do convolution."""

import numpy as np
import tvm
from tvm import autotvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

from common import get_all_backend

def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1, add_bias=False, add_relu=False):
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))

    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    bias = tvm.placeholder((num_filter, 1, 1), name='bias')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nchw.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.nn.conv2d(A, W, (stride, stride), (padding, padding),
                               (dilation, dilation), layout='NCHW', out_dtype=dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.generic.schedule_conv2d_nchw([C])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            func = tvm.build(s, [A, W, bias, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4)

    for device in get_all_backend():
        with autotvm.tophub.context(device):  # load tophub pre-tuned parameters
            check_device(device)


def test_conv2d_nchw():
    # ResNet18 workloads
    verify_conv2d_nchw(1,   3, 224,  64, 7, 2, 3)
    verify_conv2d_nchw(1,  64,  56,  64, 3, 1, 1)
    verify_conv2d_nchw(1,  64,  56,  64, 1, 1, 0)
    verify_conv2d_nchw(1,  64,  56, 128, 3, 2, 1)
    verify_conv2d_nchw(1,  64,  56, 128, 1, 2, 0)
    verify_conv2d_nchw(1, 128,  28, 128, 3, 1, 1)
    verify_conv2d_nchw(1, 128,  28, 256, 3, 2, 1)
    verify_conv2d_nchw(1, 128,  28, 256, 1, 2, 0)
    verify_conv2d_nchw(1, 256,  14, 256, 3, 1, 1)
    verify_conv2d_nchw(1, 256,  14, 512, 3, 2, 1)
    verify_conv2d_nchw(1, 256,  14, 512, 1, 2, 0)
    verify_conv2d_nchw(1, 512,   7, 512, 3, 1, 1)

    # bias, relu
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, add_relu=True)
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, add_bias=True)
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, add_bias=True, add_relu=True)

    # dilation = 2
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1, dilation=2)

    # batch size
    verify_conv2d_nchw(4, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nchw(9, 64, 56, 64, 3, 1, 1)

    # weird workloads
    verify_conv2d_nchw(2, 2, 2, 2, 2, 2, 2)
    verify_conv2d_nchw(3, 3, 3, 3, 3, 3, 3)
    verify_conv2d_nchw(4, 4, 4, 4, 4, 4, 4)
    verify_conv2d_nchw(5, 5, 5, 5, 5, 5, 5)
    verify_conv2d_nchw(6, 6, 6, 6, 6, 6, 6)

    # disable these tests due to some bugs of llvm with nvptx
    # verify_conv2d_nchw(1, 1, 1, 1, 1, 1, 1, dilation=1)
    # verify_conv2d_nchw(1, 1, 1, 1, 1, 1, 1, dilation=2)
    # verify_conv2d_nchw(2, 13, 71, 59, 3, 1, 1)

    # inception v3 workloads
    verify_conv2d_nchw(1,    3, 299,  32, 3, 2, 0)
    verify_conv2d_nchw(1,   32, 149,  32, 3, 1, 0)
    verify_conv2d_nchw(1,   32, 147,  64, 3, 1, 1)
    verify_conv2d_nchw(1,   64,  73,  80, 1, 1, 0)
    verify_conv2d_nchw(1,   80,  73, 192, 3, 1, 0)
    verify_conv2d_nchw(1,  192,  35,  64, 1, 1, 0)
    verify_conv2d_nchw(1,  192,  35,  48, 1, 1, 0)
    verify_conv2d_nchw(1,   48,  35,  64, 5, 1, 2)
    verify_conv2d_nchw(1,   64,  35,  96, 3, 1, 1)
    verify_conv2d_nchw(1,   96,  35,  96, 3, 1, 1)
    verify_conv2d_nchw(1,  192,  35,  32, 1, 1, 0)
    verify_conv2d_nchw(1,  256,  35,  64, 1, 1, 0)
    verify_conv2d_nchw(1,  256,  35,  48, 1, 1, 0)
    verify_conv2d_nchw(1,  288,  35,  64, 1, 1, 0)
    verify_conv2d_nchw(1,  288,  35,  48, 1, 1, 0)
    verify_conv2d_nchw(1,  288,  35, 384, 3, 2, 0)
    verify_conv2d_nchw(1,   96,  35,  96, 3, 2, 0)
    verify_conv2d_nchw(1,  768,  17, 192, 1, 1, 0)
    verify_conv2d_nchw(1,  768,  17, 128, 1, 1, 0)
    verify_conv2d_nchw(1,  128,  17, 128, 1, 1, 0)
    verify_conv2d_nchw(1,  128,  17, 192, 7, 1, 3)
    verify_conv2d_nchw(1,  128,  17, 128, 7, 1, 3)
    verify_conv2d_nchw(1,  128,  17, 192, 1, 1, 0)
    verify_conv2d_nchw(1,  768,  17, 160, 1, 1, 0)
    verify_conv2d_nchw(1,  160,  17, 160, 1, 1, 0)
    verify_conv2d_nchw(1,  160,  17, 192, 7, 1, 3)
    verify_conv2d_nchw(1,  160,  17, 160, 7, 1, 3)
    verify_conv2d_nchw(1,  160,  17, 192, 1, 1, 0)
    verify_conv2d_nchw(1,  192,  17, 192, 1, 1, 0)
    verify_conv2d_nchw(1,  192,  17, 192, 7, 1, 3)
    verify_conv2d_nchw(1,  192,  17, 320, 3, 2, 0)
    verify_conv2d_nchw(1,  192,  17, 192, 3, 2, 0)
    verify_conv2d_nchw(1, 1280,   8, 320, 1, 1, 0)
    verify_conv2d_nchw(1, 1280,   8, 384, 1, 1, 0)
    verify_conv2d_nchw(1,  384,   8, 384, 1, 1, 0)
    verify_conv2d_nchw(1,  384,   8, 384, 3, 1, 1)
    verify_conv2d_nchw(1, 1280,   8, 448, 1, 1, 0)
    verify_conv2d_nchw(1,  448,   8, 384, 3, 1, 1)
    verify_conv2d_nchw(1, 1280,   8, 192, 1, 1, 0)
    verify_conv2d_nchw(1, 2048,   8, 320, 1, 1, 0)
    verify_conv2d_nchw(1, 2048,   8, 384, 1, 1, 0)
    verify_conv2d_nchw(1, 2048,   8, 448, 1, 1, 0)
    verify_conv2d_nchw(1, 2048,   8, 192, 1, 1, 0)
    verify_conv2d_nchw(1, 1024,  19,  84, 3, 1, 1)
    verify_conv2d_nchw(1, 2048,  10, 126, 3, 1, 1)
    verify_conv2d_nchw(1,  512,   5, 126, 3, 1, 1)
    verify_conv2d_nchw(1,  256,   3, 126, 3, 1, 1)


if __name__ == "__main__":
    test_conv2d_nchw()
