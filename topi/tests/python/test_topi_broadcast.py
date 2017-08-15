"""Test code for broadcasting operators."""
import os
import numpy as np
import tvm
import topi

def verify_broadcast_to_ele(in_shape, out_shape):
    # Build the logic and compile the function
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.broadcast_to(A, out_shape)
    s = topi.cuda.schedule_broadcast_to(B)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        foo = tvm.build(s, [A, B], device, name="broadcast_to")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = np.broadcast_to(data_npy, out_shape)

        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), ctx)
        for _ in range(1):
            foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")


def test_broadcast_to():
    verify_broadcast_to_ele((1,), (10,))
    verify_broadcast_to_ele((1, 1, 5, 4), (3, 4, 4, 4, 5, 4))
    verify_broadcast_to_ele((1, 128, 1, 32), (64, 128, 64, 32))

if __name__ == "__main__":
    test_broadcast_to()
