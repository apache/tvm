"""Example code to do convolution."""
import os
import numpy as np
import scipy.signal
import tvm
from tvm.contrib import nvcc_compiler
import topi
from topi.nn.util import get_const_tuple

TASK = "conv2d_hwcn_map"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc_compiler.compile_source(code, target="ptx", options=["-arch=sm_52"])
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


def conv2d_hwcn_python(a_np, w_np, stride, padding):
    in_height, in_width, in_channel, batch = a_np.shape
    kernel_h, kernel_w, channel, num_filter = w_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == 'VALID':
        pad_h = 0
        pad_w = 0
    else: # 'same'
        pad_h = kernel_h - 1
        pad_w = kernel_w - 1
    pad_top = int(np.ceil(float(pad_h) / 2))
    pad_bottom = pad_h - pad_top
    pad_left = int(np.ceil(float(pad_w) / 2))
    pad_right = pad_w - pad_left
    # compute the output shape
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    # change the layout from HWCN to NCHW
    at = a_np.transpose((3, 2, 0, 1))
    wt = w_np.transpose((3, 2, 0, 1))
    bt = np.zeros((batch, out_channel, out_height, out_width))
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w))
                    apad[pad_top:-pad_bottom, pad_left:-pad_right] = at[n, c]
                else:
                    apad = at[n, c]
                out = scipy.signal.convolve2d(
                    apad, np.rot90(np.rot90(wt[f, c])), mode='valid')
                bt[n, f] += out[::stride,::stride]
    return bt.transpose((2, 3, 1, 0))


def verify_conv2d_hwcn_map(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((in_height, in_width, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    B = topi.nn.conv2d_hwcn(A, W, stride, padding)
    C = topi.nn.relu(B)
    s1 = topi.cuda.schedule_conv2d_hwcn_map(B.op)
    s2 = topi.cuda.schedule_conv2d_hwcn_map(C.op)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    w_np = np.random.uniform(size=get_const_tuple(W.shape)).astype(W.dtype)
    b_np = conv2d_hwcn_python(a_np, w_np, stride, padding)
    c_np = np.maximum(b_np, 0)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        with tvm.build_config(auto_unroll_max_step=32,
                              auto_unroll_min_depth=0,
                              unroll_explicit=False):
            func1 = tvm.build(s1, [A, W, B], device)
            func2 = tvm.build(s2, [A, W, C], device)
            func1(a, w, b)
            func2(a, w, c)
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ['cuda', 'opencl', 'metal']:
        check_device(device)


def test_conv2d_hwcn_map():
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "SAME")
    verify_conv2d_hwcn_map(4, 128, 16, 128, 5, 2, "SAME")
    verify_conv2d_hwcn_map(4, 128, 16, 256, 5, 2, "SAME")
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_hwcn_map(1, 256, 32, 256, 3, 1, "VALID")
    verify_conv2d_hwcn_map(4, 128, 16, 128, 5, 2, "VALID")
    verify_conv2d_hwcn_map(4, 128, 16, 256, 5, 2, "VALID")


if __name__ == "__main__":
    test_conv2d_hwcn_map()
