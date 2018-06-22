"""Test code for l2 normalization"""
import numpy as np
import tvm
import topi
import logging
from topi.util import get_const_tuple
import topi.testing

def verify_l2_normalize(shape, eps, axis=None):
    '''Verify l2 normalization operator by comparing outputs from tvm and numpy implementation'''
    A = tvm.placeholder(shape, name='A')
    B = topi.cpp.nn.l2_normalize(A, eps, axis)
    dtype = A.dtype

    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = topi.testing.l2_normalize_python(a_np, eps, axis)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.default_schedule(target, [B], False)
        else:
            s = topi.cpp.cuda.schedule_l2_normalize(target, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, B], device, name="l2_normalize")
        func(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'llvm']:
        check_device(device)

def test_l2_normalize():
    verify_l2_normalize((1, 3, 20, 20), 0.001)
    verify_l2_normalize((1, 3, 20, 20), 0.001, (1,))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (1, 2))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (2, 3))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (0, 3))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (0, 2, 3))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_l2_normalize()
