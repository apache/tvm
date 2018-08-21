"""Test code for transposed convolution."""
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

from common import get_all_backend

def verify_conv2d_transpose_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((in_channel, num_filter, kernel, kernel), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_transpose.verify_conv2d_transpose_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_transpose_nchw_python(a_np, w_np, stride, padding)
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
            B = topi.nn.conv2d_transpose_nchw(A, W, [stride, stride], [padding, padding], A.dtype)
            C = topi.nn.relu(B)
            s1 = topi.generic.schedule_conv2d_transpose_nchw([B])
            s2 = topi.generic.schedule_conv2d_transpose_nchw([C])
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)

        func1 = tvm.build(s1, [A, W, B], device)
        func2 = tvm.build(s2, [A, W, C], device)
        func1(a, w, b)
        func2(a, w, c)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)


def test_conv2d_transpose_nchw():
    verify_conv2d_transpose_nchw(1, 3, 224, 32, 3, 1, 0)
    verify_conv2d_transpose_nchw(1, 3, 224, 32, 3, 2, 1)
    verify_conv2d_transpose_nchw(1, 32, 32, 128, 5, 1, 0)
    verify_conv2d_transpose_nchw(1, 32, 32, 128, 5, 2, 1)


if __name__ == "__main__":
    test_conv2d_transpose_nchw()
