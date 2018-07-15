"""Example code to do region."""
import numpy as np
import topi
from topi.util import get_const_tuple
import tvm
import topi.testing

def verify_region(batch, in_size, in_channel, n, classes, coords, background, l_softmax):
    '''Verify region operator by comparing outputs from tvm and numpy implementation'''
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.vision.yolo.region(A, n, classes, coords, background, l_softmax)

    a_shape = get_const_tuple(A.shape)
    dtype = A.dtype

    def get_ref_data_region():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_np = topi.testing.region_python(a_np, n, classes, coords, background, l_softmax)
        return a_np, b_np

    a_np, b_np = get_ref_data_region()
    def check_device(device):
        '''Cheching devices is enabled or not'''
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                s = topi.generic.vision.schedule_region([B])
            else:
                s = topi.cuda.vision.schedule_region([B])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, B], device)
        func(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda']:
        check_device(device)

def test_region():
    verify_region(1, 19, 425, 5, 80, 4, 0, 1)

if __name__ == "__main__":
    test_region()
