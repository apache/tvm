"""Example code to do convolution."""
import tvm
import os
from tvm.contrib import nvcc_compiler
import numpy as np
import scipy.signal

TASK="conv"
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

def conv_gpu(tensorA, tensorW, tensorB, stride=1, pad=0, device='cuda'):
    """Compute the convolution in GPU

    Parameters
    ----------
    tensorA : NDArray
        Input tensor, layout is HWCN

    tensorW : NDArray
        Weight tensor, layout is kernel x kernel x channel x num_filter

    tensorB : NDArray
        Output tensor, layout is HWCN

    stride : int
        Stride

    pad : int
        Padding

    device : ['cuda', 'opencl']
        Device name
    """
    in_height, in_width, in_channel, batch = tensorA.shape
    kernel, kernel, channel, num_filter = tensorW.shape
    assert device in ['cuda', 'opencl']
    if not tvm.module.enabled(device):
        raise RuntimeError("Device %s is not enabled" % device)

    # graph
    A = tvm.placeholder(tensorA.shape, name='A')
    W = tvm.placeholder(tensorW.shape, name='W')
    Apad = tvm.compute(
        (in_height + pad * 2, in_width + pad * 2, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.select(
            tvm.all(yy >= pad, yy - pad < in_height,
                    xx >= pad, xx - pad < in_width),
            A[yy - pad, xx - pad, cc, nn], tvm.const(0.)))

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel), name='ry')
    rx = tvm.reduce_axis((0, kernel), name='rx')

    B = tvm.compute(
        tensorB.shape,
        lambda yy, xx, ff, nn: tvm.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name='B')

    # schedule
    s = tvm.create_schedule(B.op)
    s[Apad].compute_inline()
    AA = s.cache_read(Apad, "shared", [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BB = s.cache_write(B, "local")

    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2
    
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    hi, wi, fi, ni = s[B].op.axis
    bz = s[B].fuse(wi, hi)
    by, fi = s[B].split(fi, factor=block_factor)
    bx, ni = s[B].split(ni, factor=block_factor)
    tyz, fi = s[B].split(fi, nparts=vthread)
    txz, ni = s[B].split(ni, nparts=vthread)
    ty, fi = s[B].split(fi, nparts=num_thread)
    tx, ni = s[B].split(ni, nparts=num_thread)
    s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
    s[B].bind(bz, block_z)
    s[B].bind(by, block_y)
    s[B].bind(bx, block_x)
    s[B].bind(tyz, thread_yz)
    s[B].bind(txz, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)

    s[BB].compute_at(s[B], tx)
    yi, xi, fi, ni = s[BB].op.axis
    rco, rci = s[BB].split(rc, factor=step)
    s[BB].reorder(rco, ry, rx, rci, fi, ni)
    fuse_index = s[BB].fuse(rx, ry)
    fuse_index = s[BB].fuse(fuse_index, rco)
    rx = fuse_index

    s[AA].compute_at(s[BB], rx)
    s[WW].compute_at(s[BB], rx)
    s[AL].compute_at(s[BB], rci)
    s[WL].compute_at(s[BB], rci)
    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ni)
    # Schedule for W's shared memory load
    yi, xi, ci, fi = s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=num_thread)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    _, fi = s[WW].split(fi, factor=4)
    s[WW].reorder(ty, tx, yi, xi, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    s[WW].vectorize(fi)
    # display the IR
    # print(tvm.lower(s, [A, W, B], simple_mode=True))

    with tvm.build_config(auto_unroll_max_step=32,
                          auto_unroll_min_depth=0,
                          unroll_explicit=False,
                          detect_global_barrier=False):
        # build the kernel
        func = tvm.build(s, [A, W, B], device)
        # launch the kernel
        func(tensorA, tensorW, tensorB)

def conv_python(a_np, w_np, stride, pad):
    in_height, in_width, in_channel, batch = a_np.shape
    kernel, kernel, channel, num_filter = w_np.shape
    out_channel = num_filter
    out_height = (in_height - kernel + pad * 2) / stride + 1
    out_width = (in_width - kernel + pad * 2) / stride + 1

    at = a_np.transpose((3, 2, 0, 1))
    wt = w_np.transpose((3, 2, 0, 1))
    bt = np.zeros((batch, out_channel, out_height, out_width))

    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad > 0:
                    apad = np.zeros((in_height + pad * 2, in_width + pad *2))
                    apad[pad:-pad, pad:-pad] = at[n, c]
                else:
                    apad = at[n, c]
                out = scipy.signal.convolve2d(
                    apad, np.rot90(np.rot90(wt[f, c])), mode='valid')
                bt[n, f] += out[::stride,::stride]
    return bt.transpose((2, 3, 1, 0))

def test_conv():
    batch = 64
    in_channel = 128
    in_height = 16
    in_width = 16
    num_filter = 128
    kernel = 3
    stride = 2
    pad = 1

    out_channel = num_filter
    out_height = (in_height - kernel + pad * 2) / stride + 1
    out_width = (in_width - kernel + pad * 2) / stride + 1

    a_np = np.random.uniform(size=(in_height, in_width, in_channel, batch)).astype(np.float32)
    w_np = np.random.uniform(size=(kernel, kernel, in_channel, num_filter)).astype(np.float32)
    b_np = conv_python(a_np, w_np, stride, pad)

    for device in ['cuda', 'opencl']:
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            continue
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros((out_height, out_width, out_channel, batch),
                                  dtype=np.float32), ctx)
        conv_gpu(a, w, b, stride=stride, pad=pad, device=device)
        # check correctness
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
    
if __name__ == "__main__":
    test_conv()
