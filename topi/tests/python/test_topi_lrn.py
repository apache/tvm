"""Test code for local response normalization"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple

np.set_printoptions(threshold=np.nan)

def verify_lrn(n, c, h, w, size, bias, alpha, beta):

    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.lrn_nchw(A, size, alpha, beta, bias)
    dtype = A.dtype

    a_np = np.random.uniform(size=(n, c, h, w)).astype(dtype)
    b_np = np.zeros(shape=(n, c, h, w)).astype(dtype)
    b_np = topi.testing.lrn_nchw_python(a_np, size, bias, alpha, beta, b_np)

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
        np.testing.assert_allclose(b.asnumpy(), b_np.asnumpy(), rtol=1e-1)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)

def test_lrn():
    verify_lrn(1, 3, 5, 5, 3, 1, 1, 0.5)
    verify_lrn(1, 3, 20, 20, 3, 2, 1, 0.75)


if __name__ == "__main__":
    test_lrn()
