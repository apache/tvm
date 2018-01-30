"""Test code for binary neural network operators."""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize


def verify_binary_dense(batch, in_dim, out_dim):
    A = tvm.placeholder((batch, in_dim), name='A')
    B = tvm.placeholder((out_dim, in_dim), name='B')
    bnn_A = topi.cpp.nn.binarize_pack(A, 1)
    bnn_B = topi.cpp.nn.binarize_pack(B, 1)
    # binary dense
    bnn_A1 = tvm.placeholder(bnn_A.shape, dtype=bnn_A.dtype)
    bnn_B1 = tvm.placeholder(bnn_B.shape, dtype=bnn_B.dtype)
    bnn_C = topi.cpp.nn.binary_dense(bnn_A1, bnn_B1)
    # schedule
    target = topi.cpp.TEST_create_target("llvm")
    s1 = topi.cpp.x86.schedule_binarize_pack(target, [bnn_A])
    s2 = topi.cpp.x86.schedule_binarize_pack(target, [bnn_B])
    s3 = topi.cpp.x86.schedule_binary_dense(target, [bnn_C])

    dtype = A.dtype
    @memoize("topi.tests.test_topi_binary_dense")
    def get_ref_data():
        # generate random matrix of +1 or -1 value
        a_np = (np.random.randint(2, size=(batch, in_dim)) * 2 - 1).astype(dtype)
        b_np = (np.random.randint(2, size=(out_dim, in_dim)) * 2 - 1).astype(dtype)
        c_np = np.dot(a_np, b_np.T)
        return (a_np, b_np, c_np)

    a_np, b_np, c_np = get_ref_data()

    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    bnn_a = tvm.nd.array(np.zeros(get_const_tuple(bnn_A.shape), dtype=bnn_A.dtype), ctx)
    bnn_b = tvm.nd.array(np.zeros(get_const_tuple(bnn_B.shape), dtype=bnn_B.dtype), ctx)
    bnn_c = tvm.nd.array(np.zeros(get_const_tuple(bnn_C.shape), dtype=bnn_C.dtype), ctx)
    f1 = tvm.build(s1, [A, bnn_A], 'llvm')
    f2 = tvm.build(s2, [B, bnn_B], 'llvm')
    f3 = tvm.build(s3, [bnn_A1, bnn_B1, bnn_C], 'llvm')
    f1(a, bnn_a)
    f2(b, bnn_b)
    f3(bnn_a, bnn_b, bnn_c)
    np.testing.assert_allclose(bnn_c.asnumpy(), c_np, rtol=1e-5)

def test_binary_dense():
    verify_binary_dense(1, 4096, 1024)
    verify_binary_dense(1, 1024, 1000)


if __name__ == "__main__":
    test_binary_dense()
