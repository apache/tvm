"""Test code for pooling"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def verify_global_avg_pool(n, c, h, w):
    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.global_avg_pool(A)
    s = topi.cuda.schedule_global_avg_pool(B)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = np.mean(a_np, axis=(2,3), keepdims=True)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        f = tvm.build(s, [A, B], device, name="global_avg_pool")
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal']:
        check_device(device)

def test_global_avg_pool():
    verify_global_avg_pool(1, 256, 3, 3)
    verify_global_avg_pool(4, 256, 7, 3)
    verify_global_avg_pool(1, 1024, 7, 7)
    verify_global_avg_pool(4, 1024, 7, 7)


if __name__ == "__main__":
    test_global_avg_pool()
