"""Test code for dense operator"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize


def verify_dense(batch, in_dim, out_dim, use_bias=True):
    A = tvm.placeholder((batch, in_dim), name='A')
    B = tvm.placeholder((out_dim, in_dim), name='B')
    C = tvm.placeholder((out_dim,), name='C')
    D = topi.nn.dense(A, B, C if use_bias else None)
    D = topi.nn.relu(D)
    dtype = A.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim)).astype(dtype)
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype(dtype)
        c_np = np.random.uniform(size=(out_dim,)).astype(dtype)
        if use_bias:
            d_np = np.maximum(np.dot(a_np, b_np.T) + c_np, 0.0)
        else:
            d_np = np.maximum(np.dot(a_np, b_np.T), 0.0)
        return (a_np, b_np, c_np, d_np)
    # get the test data
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_dense(D)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B, C, D], device, name="dense")
        f(a, b, c, d)
        np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)

def test_dense():
    verify_dense(1, 1024, 1000, use_bias=True)
    verify_dense(1, 1024, 1000, use_bias=False)


if __name__ == "__main__":
    test_dense()
