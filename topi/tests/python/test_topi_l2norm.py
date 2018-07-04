"""Test code for L2 normalization"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple
import topi.testing

def verify_l2_normalize(ishape, eps, axis=None):

    A = tvm.placeholder(ishape, name='A')
    B = topi.nn.l2_normalize(A, eps, axis)
    dtype = A.dtype

    a_np = np.random.uniform(size=ishape).astype(dtype)
    b_np = topi.testing.l2_normalize_python(a_np, eps, axis)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                s = topi.generic.schedule_l2_normalize([B])
            else:
                s = topi.cuda.schedule_l2_normalize([B])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx']:
        check_device(device)

def test_l2_normalize():
    verify_l2_normalize((1, 3, 20, 20), 0.001)
    verify_l2_normalize((1, 3, 20, 20), 0.001, (1,))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (1, 2))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (2, 3))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (0, 3))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (0, 2, 3))


if __name__ == "__main__":
    test_l2_normalize()
