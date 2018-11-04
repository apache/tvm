"""Test for NCHW[x]c convolution"""

import numpy as np
import tvm
from tvm import autotvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

from common import get_all_backend

def _transform_data(data, bn):
    # NCHW -> NCHW[x]c
    batch_size, channel, height, width = data.shape
    data = np.reshape(data, (batch_size, channel//bn, bn, height, width))
    data = np.transpose(data, (0, 1, 3, 4, 2))
    return data

def _transform_kernel(kernel, ic_bn, oc_bn):
    # OIHW -> OIHW[x]i[x]o
    out_channel, in_channel, kh, kw = kernel.shape
    kernel = np.reshape(kernel, (out_channel//oc_bn, oc_bn, in_channel//ic_bn, ic_bn, kh, kw))
    kernel = np.transpose(kernel, (0, 2, 4, 5, 3, 1))
    return kernel

def _transform_bias(bias, bn):
    # [num_filter, 1, 1] -> [num_filter//bn, 1, 1, bn]
    num_filter, h, w = bias.shape
    bias = np.reshape(bias, (num_filter//bn, bn, h, w))
    bias = np.transpose(bias, (0, 2, 3, 1))
    return bias

def verify_conv2d_NCHWc(batch, in_channel, in_size, num_filter, kernel, stride,
                        padding, dilation=1, add_bias=False, add_relu=False, dtype="float32"):
    assert dilation == 1, "conv2d_NCHWc does not support dilation for now."
    print("Workload: (%d, %d, %d, %d, %d, %d, %d)" %
          (batch, in_channel, in_size, num_filter, kernel, stride, padding))

    in_height = in_width = in_size

    # for testing functionality,
    # we choose arbitrary block size that can divide the channel,
    # regardless of the performance.
    oc_block = 1
    for bn in range(16, 0, -1):
        if num_filter % bn == 0:
            oc_block = bn
            break

    ic_block = 1
    for bn in range(oc_block, 0, -1):
        if in_channel % bn == 0:
            ic_block = bn
            break

    A = tvm.placeholder((batch, in_channel//ic_block, in_height, in_width, ic_block), name='A')
    W = tvm.placeholder((num_filter//oc_block, in_channel//ic_block, kernel, kernel, ic_block, oc_block), name='W')
    bias = tvm.placeholder((num_filter//oc_block, 1, 1, oc_block), name='bias')

    @memoize("topi.tests.test_topi_conv2d_NCHWc.verify_conv2d_NCHWc")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
        w_np = np.random.uniform(size=(num_filter, in_channel, kernel, kernel)).astype(dtype)
        b_np = np.random.uniform(size=(num_filter, 1, 1)).astype(dtype)
        c_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        if add_bias:
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return _transform_data(a_np, ic_block), _transform_kernel(w_np, ic_block, oc_block), \
               _transform_bias(b_np, oc_block), _transform_data(c_np, oc_block)

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.nn.conv2d_NCHWc(A, W, (stride, stride), (padding, padding),
                                     (dilation, dilation),
                                     layout='NCHW%dc'%ic_block,
                                     out_layout="NCHW%dc"%oc_block,
                                     out_dtype=dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.generic.schedule_conv2d_NCHWc([C])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            func = tvm.build(s, [A, W, bias, C], device,
                             name="relu_%d_%d_%d_%d_%d_%d_%d_%d" %
                                  (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device,
                             name="relu_%d_%d_%d_%d_%d_%d_%d_%d" %
                                  (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    # test llvm only for now since conv2d_NCHWc implement is missing in other backend.
    for device in ["llvm"]:
        with autotvm.tophub.context(device):  # load tophub pre-tuned parameters
            check_device(device)


def test_conv2d_NCHWc():
    # ResNet18 workloads
    verify_conv2d_NCHWc(1,   3, 224,  64, 7, 2, 3)
    verify_conv2d_NCHWc(1,  64,  56,  64, 3, 1, 1)
    verify_conv2d_NCHWc(1,  64,  56,  64, 1, 1, 0)
    verify_conv2d_NCHWc(1,  64,  56, 128, 3, 2, 1)
    verify_conv2d_NCHWc(1,  64,  56, 128, 1, 2, 0)
    verify_conv2d_NCHWc(1, 128,  28, 128, 3, 1, 1)
    verify_conv2d_NCHWc(1, 128,  28, 256, 3, 2, 1)
    verify_conv2d_NCHWc(1, 128,  28, 256, 1, 2, 0)
    verify_conv2d_NCHWc(1, 256,  14, 256, 3, 1, 1)
    verify_conv2d_NCHWc(1, 256,  14, 512, 3, 2, 1)
    verify_conv2d_NCHWc(1, 256,  14, 512, 1, 2, 0)
    verify_conv2d_NCHWc(1, 512,   7, 512, 3, 1, 1)

    # bias, relu
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, add_relu=True)
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, add_bias=True)
    verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, add_bias=True, add_relu=True)

    # disable dilation test since it is not supported by NCHW[x]c conv for now.
    # verify_conv2d_NCHWc(1, 64, 56, 64, 3, 1, 1, dilation=2)

    # batch size
    verify_conv2d_NCHWc(4, 64, 56, 64, 3, 1, 1)
    verify_conv2d_NCHWc(9, 64, 56, 64, 3, 1, 1)

    # weird workloads
    verify_conv2d_NCHWc(2, 2, 2, 2, 2, 2, 2)
    verify_conv2d_NCHWc(3, 3, 3, 3, 3, 3, 3)
    verify_conv2d_NCHWc(4, 4, 4, 4, 4, 4, 4)
    verify_conv2d_NCHWc(5, 5, 5, 5, 5, 5, 5)
    verify_conv2d_NCHWc(6, 6, 6, 6, 6, 6, 6)

    # disable these tests due to some bugs of llvm with nvptx
    # verify_conv2d_NCHWc(1, 1, 1, 1, 1, 1, 1, dilation=1)
    # verify_conv2d_NCHWc(1, 1, 1, 1, 1, 1, 1, dilation=2)
    # verify_conv2d_NCHWc(2, 13, 71, 59, 3, 1, 1)

    # inception v3 workloads
    verify_conv2d_NCHWc(1,    3, 299,  32, 3, 2, 0)
    verify_conv2d_NCHWc(1,   32, 149,  32, 3, 1, 0)
    verify_conv2d_NCHWc(1,   32, 147,  64, 3, 1, 1)
    verify_conv2d_NCHWc(1,   64,  73,  80, 1, 1, 0)
    verify_conv2d_NCHWc(1,   80,  73, 192, 3, 1, 0)
    verify_conv2d_NCHWc(1,  192,  35,  64, 1, 1, 0)
    verify_conv2d_NCHWc(1,  192,  35,  48, 1, 1, 0)
    verify_conv2d_NCHWc(1,   48,  35,  64, 5, 1, 2)
    verify_conv2d_NCHWc(1,   64,  35,  96, 3, 1, 1)
    verify_conv2d_NCHWc(1,   96,  35,  96, 3, 1, 1)
    verify_conv2d_NCHWc(1,  192,  35,  32, 1, 1, 0)
    verify_conv2d_NCHWc(1,  256,  35,  64, 1, 1, 0)
    verify_conv2d_NCHWc(1,  256,  35,  48, 1, 1, 0)
    verify_conv2d_NCHWc(1,  288,  35,  64, 1, 1, 0)
    verify_conv2d_NCHWc(1,  288,  35,  48, 1, 1, 0)
    verify_conv2d_NCHWc(1,  288,  35, 384, 3, 2, 0)
    verify_conv2d_NCHWc(1,   96,  35,  96, 3, 2, 0)
    verify_conv2d_NCHWc(1,  768,  17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1,  768,  17, 128, 1, 1, 0)
    verify_conv2d_NCHWc(1,  128,  17, 128, 1, 1, 0)
    verify_conv2d_NCHWc(1,  128,  17, 192, 7, 1, 3)
    verify_conv2d_NCHWc(1,  128,  17, 128, 7, 1, 3)
    verify_conv2d_NCHWc(1,  128,  17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1,  768,  17, 160, 1, 1, 0)
    verify_conv2d_NCHWc(1,  160,  17, 160, 1, 1, 0)
    verify_conv2d_NCHWc(1,  160,  17, 192, 7, 1, 3)
    verify_conv2d_NCHWc(1,  160,  17, 160, 7, 1, 3)
    verify_conv2d_NCHWc(1,  160,  17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1,  192,  17, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1,  192,  17, 192, 7, 1, 3)
    verify_conv2d_NCHWc(1,  192,  17, 320, 3, 2, 0)
    verify_conv2d_NCHWc(1,  192,  17, 192, 3, 2, 0)
    verify_conv2d_NCHWc(1, 1280,   8, 320, 1, 1, 0)
    verify_conv2d_NCHWc(1, 1280,   8, 384, 1, 1, 0)
    verify_conv2d_NCHWc(1,  384,   8, 384, 1, 1, 0)
    verify_conv2d_NCHWc(1,  384,   8, 384, 3, 1, 1)
    verify_conv2d_NCHWc(1, 1280,   8, 448, 1, 1, 0)
    verify_conv2d_NCHWc(1,  448,   8, 384, 3, 1, 1)
    verify_conv2d_NCHWc(1, 1280,   8, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048,   8, 320, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048,   8, 384, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048,   8, 448, 1, 1, 0)
    verify_conv2d_NCHWc(1, 2048,   8, 192, 1, 1, 0)
    verify_conv2d_NCHWc(1, 1024,  19,  84, 3, 1, 1)
    verify_conv2d_NCHWc(1, 2048,  10, 126, 3, 1, 1)
    verify_conv2d_NCHWc(1,  512,   5, 126, 3, 1, 1)
    verify_conv2d_NCHWc(1,  256,   3, 126, 3, 1, 1)

if __name__ == "__main__":
    test_conv2d_NCHWc()