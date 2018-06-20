"""Test code for L2 norm"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def l2norm_instance_python(a_np, eps, axis=None):
    """L2 norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    eps : float
        epsilon constant value
    axis : list of int
        axis over the normalization applied

    Returns
    -------
    l2norm_out : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, axis1, axis2, axis3 = a_np.shape
    sqr_sum = np.zeros(shape=(batch,)).astype(a_np.dtype)
    sqrt_sum = np.zeros(shape=(batch,)).astype(a_np.dtype)
    l2norm_out = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    dot_value = np.power(a_np, 2.0)
    sqr_sum = np.sum(dot_value, axis, keepdims=True)
    sqrt_sum = np.sqrt(np.maximum(np.broadcast_to(sqr_sum, a_np.shape), eps))
    return np.divide(a_np, sqrt_sum)

def verify_l2norm(n, c, h, w, eps, axis=None):

    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.l2norm_instance(A, eps, axis)
    dtype = A.dtype

    a_np = np.random.uniform(size=(n, c, h, w)).astype(dtype)
    b_np = l2norm_instance_python(a_np, eps, axis)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_l2norm(B)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)

def test_l2norm():
    verify_l2norm(1, 3, 20, 20, 0.001)
    verify_l2norm(1, 3, 20, 20, 0.001, 1)
    verify_l2norm(1, 3, 20, 20, 0.001, (1, 2))
    verify_l2norm(1, 3, 20, 20, 0.001, (2, 3))
    verify_l2norm(1, 3, 20, 20, 0.001, (0, 3))
    verify_l2norm(1, 3, 20, 20, 0.001, (0, 2, 3))


if __name__ == "__main__":
    test_l2norm()
