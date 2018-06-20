"""Test code for clip operator"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize


def verify_clip(N, a_min, a_max, dtype):
    A = tvm.placeholder((N, N), dtype=dtype, name='A')
    B = topi.cpp.clip(A, a_min, a_max)

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_clip")
    def get_ref_data():
        a_np = np.random.uniform(a_min*2, a_max*2, size=(N, N)).astype(dtype)
        b_np = np.clip(a_np, a_min, a_max)
        return a_np, b_np
    a_np, b_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        target = topi.cpp.TEST_create_target(device)
        s = topi.cpp.generic.default_schedule(target, [B], False)
        ctx = tvm.cpu(0) if device == "llvm" else tvm.gpu(0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device, name="clip")
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm']:
        check_device(device)

def test_clip():
    verify_clip(1024, -127, 127, 'int8')
    verify_clip(1024, -127, 127, 'int16')
    verify_clip(1024, -127, 127, 'float32')


if __name__ == "__main__":
    test_clip()
