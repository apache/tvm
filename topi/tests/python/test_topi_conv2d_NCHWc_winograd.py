"""Example code to do convolution."""

import numpy as np
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple


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


def verify_conv2d_NCHWc_winograd(
        batch, in_channel, in_size, num_filter, kernel,
        stride, padding, dilation=1, add_bias=False, add_relu=False, tile_size=2):
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d, ts=%d)" %
          (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, tile_size))

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

    A = tvm.placeholder((batch, in_channel // ic_block, in_height, in_width, ic_block), name='A')
    W = tvm.placeholder((num_filter // oc_block, in_channel // ic_block, kernel, kernel, ic_block, oc_block), name='W')
    bias = tvm.placeholder((num_filter // oc_block, 1, 1, oc_block), name='bias')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype
    kernel_layout = \
        "OIHW{ic_block}i{oc_block}o".format(ic_block=ic_block, oc_block=oc_block)
    layout = "NCHW{ic_block}c".format(ic_block=ic_block)
    out_layout = "NCHW{oc_block}c".format(oc_block=oc_block)

    @memoize("topi.tests.test_topi_conv2d_NCHWc_winograd.verify_conv2d_NCHWc_winograd")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
        w_np = np.random.uniform(size=(num_filter, in_channel, kernel, kernel)).astype(dtype) * 0.01
        b_np = np.random.uniform(size=(num_filter, 1, 1)).astype(dtype)
        c_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        if add_bias:
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return _transform_data(a_np, ic_block), _transform_kernel(w_np, ic_block, oc_block), \
               _transform_bias(b_np, oc_block), _transform_data(c_np, oc_block)

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device_without_weight_transform(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            WT = topi.nn.conv2d_NCHWc_winograd_weight_transform(
                W,
                tile_size=tile_size,
                kernel_layout=kernel_layout
            )
            C = topi.nn.conv2d_NCHWc_winograd_without_weight_transform(
                A, WT, (stride, stride), (padding, padding),
                (dilation, dilation),
                layout=layout,
                out_layout=out_layout,
                out_dtype=dtype,
                tile_size=tile_size)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.generic.schedule_conv2d_NCHWc_winograd_without_weight_transform([C])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        if add_bias:
            func = tvm.build(s, [A, W, bias, C], device, name="relu_bias_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, c)
        print(np.max(np.abs(((c.asnumpy() - c_np) / (np.abs(c_np) + 0.001)))))

        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    def check_device_with_weight_transform(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            C = topi.nn.conv2d_NCHWc(
                A, W, (stride, stride), (padding, padding),
                (dilation, dilation),
                layout=layout,
                out_layout=out_layout,
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
            func = tvm.build(s, [A, W, bias, C], device, name="relu_bias_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, b, c)
        else:
            func = tvm.build(s, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func(a, w, c)
        print(np.max(np.abs(((c.asnumpy() - c_np) / (np.abs(c_np) + 0.001)))))


        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    # test llvm only for now since conv2d_NCHWc_winograd is only implemented on this backend.
    for device in ['llvm']:
        check_device_with_weight_transform(device)
        check_device_without_weight_transform(device)


class WinogradFallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = 'winograd'
        self.memory[key] = cfg
        return cfg


def test_conv2d_nchw():
    autotvm.DispatchContext.current.silent = True

    with WinogradFallback():
        # resnet 18 workloads
        verify_conv2d_NCHWc_winograd(1, 64, 56, 64, 3, 1, 1)
        verify_conv2d_NCHWc_winograd(1, 128, 28, 128, 3, 1, 1, tile_size=4)
        verify_conv2d_NCHWc_winograd(1, 256, 14, 256, 3, 1, 1, tile_size=4)
        verify_conv2d_NCHWc_winograd(1, 512, 7, 512, 3, 1, 1)

        # batch size = 2
        verify_conv2d_NCHWc_winograd(2, 64, 56, 64, 3, 1, 1)

        # relu, bias
        verify_conv2d_NCHWc_winograd(2, 64, 56, 64, 3, 1, 1, add_bias=True)
        verify_conv2d_NCHWc_winograd(2, 64, 56, 64, 3, 1, 1, add_relu=True)
        verify_conv2d_NCHWc_winograd(2, 64, 56, 64, 3, 1, 1, add_relu=True, add_bias=True)

        # werid workloads
        verify_conv2d_NCHWc_winograd(1, 1, 1, 1, 3, 1, 1)
        verify_conv2d_NCHWc_winograd(3, 3, 3, 3, 3, 1, 1)
        verify_conv2d_NCHWc_winograd(2, 13, 71, 59, 3, 1, 1)

if __name__ == "__main__":
    test_conv2d_nchw()
