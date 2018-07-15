"""Test code for local response normalization"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple
import topi.testing

def verify_lrn(shape, size, axis, bias, alpha, beta):
    A = tvm.placeholder(shape, name='A')
    B = topi.nn.lrn(A, size, axis, alpha, beta, bias)
    dtype = A.dtype

    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = topi.testing.lrn_python(a_np, size, axis, bias, alpha, beta)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                s = topi.generic.schedule_lrn([B])
            else:
                s = topi.cuda.schedule_lrn([B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx']:
        check_device(device)

def test_lrn():
    verify_lrn((1, 3, 5, 5), 3, 1, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 5, 5), 3, 3, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 20, 20), 3, 1, 2.0, 1.0, 0.75)

if __name__ == "__main__":
    test_lrn()
