"""Example code to do conv2d."""
import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple


def verify_conv2d(batch, in_size, in_channel, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    with tvm.target.rasp():
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
        B = topi.nn.conv2d(A, W, stride, padding)
        s = topi.generic.schedule_conv2d_nchw([B])

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d.verify_conv2d")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()

    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, W, B], "llvm")
    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

def test_conv2d():
    verify_conv2d(1, 56,  64, 64,  3, 1, 1)

if __name__ == "__main__":
    test_conv2d()
