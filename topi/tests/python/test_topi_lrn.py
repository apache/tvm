"""Test code for local response normalization"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def lrn_nchw_python(a_np, size, bias, alpha, beta):
    """Local response norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    size : int
        normalisation window size

    bias : float
        offset to avoid dividing by 0. constant value

    alpha : float
        contant valie

    beta : float
        exponent constant value

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, channel, height, weight = a_np.shape
    radius = int(size / 2)
    sqr_sum = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    sqr_sum_up = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    lrn_out = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    def add_dot_values(i, c, j, k):
        for rxk in range(-radius, radius+1):
            if ((c + rxk) < channel) and ((c + rxk) >= 0):
                sqr_sum[i, c, j, k] = sqr_sum[i, c, j, k] + \
                                       (a_np[i, c + rxk, j, k] * \
                                        a_np[i, c + rxk, j, k])
    for i in range(batch):
        for c in range(channel):
            for j in range(height):
                for k in range(weight):
                    add_dot_values(i, c, j, k)
    for i in range(batch):
        for c in range(channel):
            for j in range(height):
                for k in range(weight):
                    sqr_sum_up[i, c, j, k] = \
                        np.power((bias + (alpha * sqr_sum[i, c, j, k] / \
                                          size)), beta)
    for i in range(batch):
        for c in range(channel):
            for j in range(height):
                for k in range(weight):
                    lrn_out[i, c, j, k] = a_np[i, c, j, k] / \
                                           sqr_sum_up[i, c, j, k]
    return lrn_out

def verify_lrn(n, c, h, w, size, bias, alpha, beta):

    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.lrn_nchw(A, size, alpha, beta, bias)
    dtype = A.dtype

    a_np = np.random.uniform(size=(n, c, h, w)).astype(dtype)
    b_np = lrn_nchw_python(a_np, size, bias, alpha, beta)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_lrn(B)
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)

def test_lrn():
    verify_lrn(1, 3, 5, 5, 3, 1, 1, 0.5)
    verify_lrn(1, 3, 20, 20, 3, 2, 1, 0.75)

if __name__ == "__main__":
    test_lrn()
