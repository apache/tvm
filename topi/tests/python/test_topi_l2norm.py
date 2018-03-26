"""Test code for L2 norm"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def l2norm_instance_python(a_np, eps):
    """L2 norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    eps : float
        epsilon constant value

    Returns
    -------
    l2norm_out : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, axis1, axis2, axis3 = a_np.shape
    sqr_sum = np.zeros(shape=(batch,)).astype(a_np.dtype)
    sqrt_sum = np.zeros(shape=(batch,)).astype(a_np.dtype)
    l2norm_out = np.zeros(shape=a_np.shape).astype(a_np.dtype)

    for i in range(batch):
        for j in range(axis1):
            for k in range(axis2):
                for m in range(axis3):
                    sqr_sum[i] = sqr_sum[i] + (a_np[i, j, k, m] * \
                                               a_np[i, j, k, m])
    for b in range(batch):
        sqrt_sum[b] = np.sqrt(sqr_sum[b] + eps)
    for b in range(batch):
        for j in range(axis1):
            for k in range(axis2):
                for m in range(axis3):
                    l2norm_out[b, j, k, m] = a_np[b, j, k, m]/sqrt_sum[b]
    return l2norm_out

def verify_l2norm(n, c, h, w, eps):

    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.l2norm_instance(A, eps)
    dtype = A.dtype

    a_np = np.random.uniform(size=(n, c, h, w)).astype(dtype)
    b_np = l2norm_instance_python(a_np, eps)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_l2norm(B)
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)

def test_l2norm():
    verify_l2norm(1, 3, 20, 20, 0.001)


if __name__ == "__main__":
    test_l2norm()
