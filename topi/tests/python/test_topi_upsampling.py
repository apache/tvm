"""Test code for upsampling"""
import numpy as np
import tvm
import topi
import math
from skimage.transform import resize

def verify_upsampling(batch, in_channel, in_height, in_width, scale):
    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.nn.upsampling(A, scale)
    out_shape = (batch, in_channel, in_height*scale, in_width*scale)
    dtype = A.dtype

    a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    b_np = np.zeros(out_shape, dtype=dtype)
    for b in range(batch):
        for c in range(in_channel):
            b_np[b,c,:,:] = resize(a_np[b,c,:,:] , (in_height*scale, in_width*scale), order=0, mode="reflect")

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_upsampling(B)
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)

        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda']:
        check_device(device)

def test_upsampling():
    verify_upsampling(8, 16, 32, 32, 2)
    verify_upsampling(12, 32, 64, 64, 3)

if __name__ == "__main__":
    test_upsampling()
