"""Test code for pooling"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

def verify_global_pool(n, c, h, w, pool_type):
    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.global_pool(A, pool_type=pool_type)
    B = topi.nn.relu(B)
    s = topi.cuda.schedule_global_pool(B)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    if pool_type == 'avg':
        b_np = np.mean(a_np, axis=(2,3), keepdims=True)
    elif pool_type =='max':
        b_np = np.max(a_np, axis=(2,3), keepdims=True)
    b_np = np.maximum(b_np, 0.0)

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

def test_global_pool():
    verify_global_pool(1, 1024, 7, 7, 'avg')
    verify_global_pool(4, 1024, 7, 7, 'avg')
    verify_global_pool(1, 1024, 7, 7, 'max')
    verify_global_pool(4, 1024, 7, 7, 'max')


if __name__ == "__main__":
    test_global_pool()
