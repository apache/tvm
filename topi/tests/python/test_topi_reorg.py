"""Example code to do reorg."""
import numpy as np
import topi
from topi.util import get_const_tuple
import tvm
from tvm.contrib.pickle_memoize import memoize


def verify_reorg(batch, in_size, in_channel, stride):
    '''Verify reorg operator by comparing outputs from tvm and numpy implementation'''
    in_height = in_width = in_size

    with tvm.target.rasp():
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        B = topi.vision.reorg(A, stride)
        s = topi.generic.schedule_reorg([B])

    a_shape = get_const_tuple(A.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_reorg.verify_reorg")
    def get_ref_data_reorg():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_np = topi.testing.reorg_python(a_np, stride)
        return a_np, b_np

    a_np, b_np = get_ref_data_reorg()

    def check_device(device):
        '''Cheching devices is enabled or not'''
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        ctx = tvm.cpu(0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, B], device)
        func(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda']:
        check_device(device)

def test_reorg():
    verify_reorg(1, 20, 8, 2)

if __name__ == "__main__":
    test_reorg()
