"""Example code to do shortcut."""
import numpy as np
import topi
from topi.util import get_const_tuple
import tvm

def verify_shortcut(batch, in_size, in_channel):
    '''Verify shortcut operator by comparing outputs from tvm and numpy implementation'''
    in_height = in_width = in_size

    A1 = tvm.placeholder((batch, in_channel, in_height, in_width), name='A1')
    A2 = tvm.placeholder((batch, in_channel, in_height, in_width), name='A2')
    B = topi.vision.shortcut(A1, A2)

    a_shape = get_const_tuple(A1.shape)
    dtype = A1.dtype
    def get_ref_data_shortcut():
        a_np1 = np.random.uniform(size=a_shape).astype(dtype)
        a_np2 = np.random.uniform(size=a_shape).astype(dtype)
        b_np = topi.testing.shortcut_python(a_np1, a_np2)
        return a_np1, a_np2, b_np

    a_np1, a_np2, b_np = get_ref_data_shortcut()
    def check_device(device):
        '''Cheching devices is enabled or not'''
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective([B])

        a1 = tvm.nd.array(a_np1, ctx)
        a2 = tvm.nd.array(a_np2, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A1, A2, B], device)
        func(a1, a2, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda']:
        check_device(device)

def test_shortcut():
    verify_shortcut(1, 144, 32)

if __name__ == "__main__":
    test_shortcut()
