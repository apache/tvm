"""Test code for L2 norm"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple

def verify_l2norm(n, c, h, w, eps):

    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.l2norm_instance_nchw(A, eps)
    dtype = A.dtype

    a_np = np.random.uniform(size=(n, c, h, w)).astype(dtype)
    b_np = topi.testing.l2norm_nchw_python(a_np, eps)

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
