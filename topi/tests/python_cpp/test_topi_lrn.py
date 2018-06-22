"""Test code for LRN"""
import numpy as np
import tvm
import topi
import logging
from topi.util import get_const_tuple
import topi.testing

def verify_lrn(shape, size, axis, bias, alpha, beta):
    '''Verify Local response normalization operator by comparing outputs from tvm and numpy implementation'''
    A = tvm.placeholder(shape, name='A')
    B = topi.cpp.nn.lrn(A, size, axis, alpha, beta, bias)
    dtype = A.dtype

    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = topi.testing.lrn_python(a_np, size, axis, bias, alpha, beta)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.default_schedule(target, [B], False)
        else:
            s = topi.cpp.cuda.schedule_lrn(target, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-1)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'llvm']:
        check_device(device)

def test_lrn():
    verify_lrn((1, 3, 5, 5), 3, 3, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 5, 5), 3, 3, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 20, 20), 3, 1, 2.0, 1.0, 0.75)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_lrn()
