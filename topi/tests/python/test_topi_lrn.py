"""Test code for local response normalization"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def lrn_python(a_np, size, axis, bias, alpha, beta):
    """Local response norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    size : int
        normalisation window size

    axis : int
        input data layout channel axis

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
    axis0, axis1, axis2, axis3 = a_np.shape
    radius = int(size / 2)
    sqr_sum = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    sqr_sum_up = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    lrn_out = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    def add_dot_values(i, j, k, l):
        if (axis == 1):
            #NCHW layout
            for rxj in range(-radius, radius+1):
                if ((j + rxj) < axis1) and ((j + rxj) >= 0):
                    sqr_sum[i, j, k, l] = sqr_sum[i, j, k, l] + \
                                           (a_np[i, j + rxj, k, l] * \
                                            a_np[i, j + rxj, k, l])
        elif (axis == 3):
            #NHWC layout
            for rxl in range(-radius, radius+1):
                if ((l + rxl) < axis3) and ((l + rxl) >= 0):
                    sqr_sum[i, j, k, l] = sqr_sum[i, j, k, l] + \
                                           (a_np[i, j, k, l + rxl] * \
                                            a_np[i, j, k, l + rxl])
    for i in range(axis0):
        for j in range(axis1):
            for k in range(axis2):
                for l in range(axis3):
                    add_dot_values(i, j, k, l)
    for i in range(axis0):
        for j in range(axis1):
            for k in range(axis2):
                for l in range(axis3):
                    sqr_sum_up[i, j, k, l] = \
                        np.power((bias + (alpha * sqr_sum[i, j, k, l] / \
                                          size)), beta)
    for i in range(axis0):
        for j in range(axis1):
            for k in range(axis2):
                for l in range(axis3):
                    lrn_out[i, j, k, l] = a_np[i, j, k, l] / \
                                           sqr_sum_up[i, j, k, l]
    return lrn_out

def verify_lrn(shape, size, axis, bias, alpha, beta):
    A = tvm.placeholder(shape, name='A')
    B = topi.nn.lrn(A, size, axis, alpha, beta, bias)
    dtype = A.dtype

    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = lrn_python(a_np, size, axis, bias, alpha, beta)

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
    verify_lrn((1, 3, 5, 5), 3, 1, 1, 1, 0.5)
    verify_lrn((1, 3, 5, 5), 3, 3, 1, 1, 0.5)
    verify_lrn((1, 3, 20, 20), 3, 1, 2, 1, 0.75)

if __name__ == "__main__":
    test_lrn()
