# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Example code to do group convolution."""
import os
import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi.testing.conv2d_nchw_python import conv2d_nchw_python
from topi.testing.conv2d_hwcn_python import conv2d_hwcn_python
from topi.testing.conv2d_nhwc_python import conv2d_nhwc_python

def grouped_conv2d_nchw_python(a_np, w_np, groups, stride, padding):
    """Grouped Convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    groups : int
        Filter groups, this indicate the number of split convolution have to
        perform

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    a_nps = np.split(a_np, groups, 1)
    w_nps = np.split(w_np, groups, 0)

    conv_out = []
    for data, kernel in zip(a_nps, w_nps):
        conv_out.append(conv2d_nchw_python(data, kernel, stride, padding))

    return np.concatenate(conv_out, 1)

def grouped_conv2d_hwcn_python(a_np, w_np, groups, stride, padding):
    """Grouped Convolution operator in HWCN layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [in_height, in_width, in_channel, batch]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    groups : int
        Filter groups, this indicate the number of split convolution have to
        perform

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [out_height, out_width, out_channel, batch]
    """

    a_nps = np.split(a_np, groups, 2)
    w_nps = np.split(w_np, groups, 3)

    conv_out = []
    for data, kernel in zip(a_nps, w_nps):
        conv_out.append(conv2d_hwcn_python(data, kernel, stride, padding))

    return np.concatenate(conv_out, 2)

def grouped_conv2d_nhwc_python(a_np, w_np, groups, stride, padding):
    """Grouped Convolution operator in NHWC layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    groups : int
        Filter groups, this indicate the number of split convolution have to
        perform

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_height,  out_width, out_channel]
    """

    a_nps = np.split(a_np, groups, 3)
    w_nps = np.split(w_np, groups, 3)

    conv_out = []
    for data, kernel in zip(a_nps, w_nps):
        conv_out.append(conv2d_nhwc_python(data, kernel, stride, padding))

    return np.concatenate(conv_out, 3)


def verify_grouped_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, groups, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    B = topi.nn.grouped_conv2d(A, W, groups, stride, padding)
    C = topi.nn.relu(B)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)
    b_np = grouped_conv2d_nchw_python(a_np, w_np, groups, stride, padding)
    c_np = np.maximum(b_np, 0)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s1 = topi.generic.schedule_grouped_conv2d([B])
            s2 = topi.generic.schedule_grouped_conv2d([C])
        ctx = tvm.context(device, 0)
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

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)


def verify_grouped_conv2d_hwcn(batch, in_channel, in_size, num_filter, kernel, groups, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((in_height, in_width, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.grouped_conv2d(A, W, groups, stride, padding, layout='HWCN')
    C = topi.nn.relu(B)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)
    b_np = grouped_conv2d_hwcn_python(a_np, w_np, groups, stride, padding)
    c_np = np.maximum(b_np, 0)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s1 = topi.generic.schedule_grouped_conv2d([B])
            s2 = topi.generic.schedule_grouped_conv2d([C])
        ctx = tvm.context(device, 0)
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

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)

def verify_grouped_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, groups, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.grouped_conv2d(A, W, groups, stride, padding, layout='NHWC')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)
    b_np = grouped_conv2d_nhwc_python(a_np, w_np, groups, stride, padding)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_conv2d_nhwc([B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm']:
        check_device(device)

def test_grouped_conv2d():
    verify_grouped_conv2d_nchw(1, 64, 56, 64, 3, 2, 1, 1)
    verify_grouped_conv2d_hwcn(1, 64, 56, 64, 3, 2, 1, 1)
    verify_grouped_conv2d_nhwc(1, 64, 56, 64, 3, 2, 1, 1)

if __name__ == "__main__":
    test_grouped_conv2d()
