"""Test code for bilinear scale """
import numpy as np
import tvm
import topi
import topi.testing
import math

def verify_bilinear_scale(batch, in_channel, in_height, in_width, out_height, out_width, layout='NCHW', align_corners=False):

    if layout == 'NCHW':
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A', dtype='float32')
        dtype = A.dtype
        out_shape = (batch, in_channel, out_height, out_width)
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    elif layout == 'NHWC':
        A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A', dtype='float32')
        dtype = A.dtype
        out_shape = (batch, out_height, out_width, in_channel)
        a_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(dtype)
    else:
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    B = topi.image.resize(A, (out_height, out_width), layout=layout, align_corners=align_corners)

    b_np = topi.testing.bilinear_resize_python(a_np, (out_height, out_width), layout, align_corners)

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

        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-3, atol=1e-3)

    for device in ['llvm', 'cuda', 'vulkan', 'nvptx']:
        check_device(device)

def test_resize():
    # Scale NCHW
    verify_bilinear_scale(4, 16, 32, 32, 50, 50, 'NCHW')
    # Scale NCHW + Align Corners
    verify_bilinear_scale(6, 32, 64, 64, 20, 20, 'NCHW', True)
    # Scale NHWC
    verify_bilinear_scale(4, 16, 32, 32, 50, 50, "NHWC")
    # Scale NHWC + Align Corners
    verify_bilinear_scale(6, 32, 64, 64, 20, 20, "NHWC", True)

if __name__ == "__main__":
    test_resize()
