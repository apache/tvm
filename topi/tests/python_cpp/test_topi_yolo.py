"""Test code for yolo op"""
import logging
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple

def verify_yolo(ishape, n, classes):
    '''Verify yolo operator by comparing outputs from tvm and numpy implementation'''
    
    A = tvm.placeholder(ishape, name='A')
    B = topi.cpp.yolo.yolo(A, n, classes)
    dtype = A.dtype

    def get_ref_data_yolo():
        '''Randomly initialize the data variables and get refernce output for the yolo operation'''
        a_np = np.random.uniform(size=ishape).astype(dtype)
        b_np = topi.testing.yolo_python(a_np, n, classes)
        return a_np, b_np

    a_np, b_np = get_ref_data_yolo()
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
        func = tvm.build(s, [A, B], device, name="yolo")
        func(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'llvm', 'vulkan']:
        check_device(device)

def test_yolo():
    verify_yolo((1, 425, 19, 19), 5, 80)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_yolo()
