"""Test code for reorg"""
import logging
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple

def verify_reorg(batch, in_size, in_channel, stride):
    '''Verify reorg operator by comparing outputs from tvm and numpy implementation'''
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    B = topi.cpp.vision.reorg(A, stride)

    a_shape = get_const_tuple(A.shape)
    dtype = A.dtype

    def get_ref_data_reorg():
        '''Randomly initialize the data variables and get refernce output for the reorg operation'''
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_np = topi.testing.reorg_python(a_np, stride)
        return a_np, b_np

    a_np, b_np = get_ref_data_reorg()
    def check_device(device):
        '''Check the device is available and if so, build and run the program'''
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.default_schedule(target, [B], False)
        else:
            s = topi.cpp.cuda.schedule_injective(target, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, B], device, name="reorg")
        func(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'llvm', 'vulkan']:
        check_device(device)

def test_reorg():
    verify_reorg(1, 38, 64, 2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_reorg()
