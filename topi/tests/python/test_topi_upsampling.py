"""Test code for upsampling"""
import numpy as np
import tvm
import topi
import topi.testing
import math

def verify_upsampling(batch, in_channel, in_height, in_width, scale):
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.nn.upsampling(A, scale)
    out_shape = (batch, in_channel, in_height*scale, in_width*scale)
    dtype = A.dtype

    a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    b_np = topi.testing.upsampling_python(a_np, scale)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)

        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda', 'vulkan']:
        check_device(device)

def test_upsampling():
    verify_upsampling(8, 16, 32, 32, 2)
    verify_upsampling(12, 32, 64, 64, 3)

if __name__ == "__main__":
    test_upsampling()
