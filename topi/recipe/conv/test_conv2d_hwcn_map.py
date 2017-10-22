"""Example code to do convolution."""
import os
import numpy as np
import scipy.signal
import tvm
from tvm.contrib import nvcc
import topi
from topi.util import get_const_tuple

TASK = "conv2d_hwcn_map"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def test_conv2d_hwcn_map():
    batch = 64
    in_channel = 128
    in_height = 16
    in_width = 16
    num_filter = 128
    kernel = 3
    stride = 2
    padding = 'SAME'

    A = tvm.placeholder((in_height, in_width, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.conv2d_hwcn(A, W, stride, padding)
    C = topi.nn.relu(B)
    s1 = topi.cuda.schedule_conv2d_hwcn([B])
    s2 = topi.cuda.schedule_conv2d_hwcn([C])

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    w_np = np.random.uniform(size=get_const_tuple(W.shape)).astype(W.dtype)
    b_np = topi.testing.conv2d_hwcn_python(a_np, w_np, stride, padding)
    c_np = np.maximum(b_np, 0)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        with tvm.build_config(auto_unroll_max_step=128,
                              unroll_explicit=device == 'rocm'):
            func1 = tvm.build(s1, [A, W, B], device)
            func1(a, w, b)
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            func2 = tvm.build(s2, [A, W, C], device)
            func2(a, w, c)
            np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'rocm']:
        check_device(device)


if __name__ == "__main__":
    test_conv2d_hwcn_map()
