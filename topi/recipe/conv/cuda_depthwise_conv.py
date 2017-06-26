"""Depthwise convolution example.

Input : 4-D tensor with shape [batch, in_channel, in_height, in_width]
Filter: 4-D tensor with shape [in_channel, channel_multiplier, filter_height, filter_width]
Stride: 1-D of size 2 

Output: 4-D tensor with shape [batch, in_channel * channel_multiplier, out_height, out_width]
"""

import tvm
import os
from tvm.contrib import nvcc_compiler
import numpy as np
from scipy import signal
import time

TASK="depthwise_conv"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx =  nvcc_compiler.compile_source(code, target="ptx", options=["-arch=sm_52"])
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

def depthwise_conv(Input, Filter, Stride):
    # graph
    in_batch = Input.shape[0]
    in_channel = Input.shape[1]
    in_height = Input.shape[2]
    in_width = Input.shape[3]

    filter_channel = Filter.shape[0]
    channel_multiplier = Filter.shape[1]
    filter_height = Filter.shape[2]
    filter_width = Filter.shape[3]

    stride_h = Stride.asnumpy()[0]
    stride_w = Stride.asnumpy()[1]

    out_batch = in_batch
    out_channel = in_channel * channel_multiplier
    out_height = (in_height - filter_height)/stride_h + 1
    out_width = (in_width - filter_width)/stride_w + 1

    # img2col
    MatrixInput = tvm.compute(
        (in_batch, in_channel, out_height*out_width, filter_height*filter_width),
        lambda bb, cc, ii, jj: Input[bb, cc, ii/out_width*stride_h + jj/filter_width, ii%out_width*stride_w + jj%filter_width],
        name="MatrixInput")
    MatrixFilter = tvm.compute(
        (filter_channel, channel_multiplier, filter_height*filter_width),
        lambda cc, mm, ii: Filter[cc, mm, ii/filter_width, ii%filter_width],
        name="MatrixFilter")
    # matrix mult and col2img
    k = tvm.reduce_axis((0, filter_height*filter_width), name='k')
    Output = tvm.compute(
        (out_batch, out_channel, out_height, out_width), 
        lambda bb, cc, ii, jj: tvm.sum(MatrixFilter[cc/channel_multiplier, cc%channel_multiplier, k]*MatrixInput[bb, cc/channel_multiplier, ii*out_width+jj, k], axis=k),
        name="Output")

    # schedule
    s = tvm.create_schedule(Output.op)
    FS = s.cache_read(Filter, "shared", [MatrixFilter]) 
    IS = s.cache_read(Input, "shared", [MatrixInput])

    s[MatrixInput].compute_inline()
    s[MatrixFilter].compute_inline()
    s[Output].unroll(k)
    
    num_thread = 512
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

    s[Output].bind(Output.op.axis[0], block_x)
    s[Output].bind(Output.op.axis[1], block_y)   
    fused = s[Output].fuse(Output.op.axis[3], Output.op.axis[2])
    tx, xi = s[Output].split(fused, nparts=num_thread)
    s[Output].bind(tx, thread_x)
    
    # schedule for Input's shared memory load
    s[IS].compute_at(s[Output], Output.op.axis[1])   
    fused = s[IS].fuse(IS.op.axis[3], IS.op.axis[2])
    tx, xi = s[IS].split(fused, nparts=num_thread)
    s[IS].bind(tx, thread_x)
    # schedule for Filter's shared memory load
    s[FS].compute_at(s[Output], Output.op.axis[1])
    fused = s[FS].fuse(FS.op.axis[3], FS.op.axis[2])
    tx, xi = s[FS].split(fused, nparts=num_thread)
    s[FS].bind(tx, thread_x)

    return s, Output

def test_depthwise_conv():
    in_batch = 2
    in_channel = 728
    in_height = 21
    in_width = 21

    filter_channel = in_channel
    channel_multiplier = 2
    filter_height = 5
    filter_width = 5

    stride_h = 1
    stride_w = 1

    Input = tvm.placeholder((in_batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Stride = tvm.nd.array(np.array([stride_h, stride_w]))

    schedule, Output = depthwise_conv(Input, Filter, Stride)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        f = tvm.build(schedule, [Input, Filter, Output], device)
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # launch the kernel
        input_np = np.random.uniform(size=(in_batch, in_channel, in_height, in_width)).astype(Input.dtype)
        filter_np = np.random.uniform(size=(in_channel, channel_multiplier, filter_height, filter_width)).astype(Filter.dtype)
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        out_batch = in_batch
        out_channel = in_channel * channel_multiplier
        out_height = (in_height - filter_height)/stride_h + 1
        out_width = (in_width - filter_width)/stride_w + 1
        output_tvm = tvm.nd.array(np.zeros((out_batch, out_channel, out_height, out_width), dtype=Output.dtype), ctx)
        # skip first pass as it is compilation
        f(input_tvm, filter_tvm, output_tvm)
        ctx.sync()
        # measure time cost of 10000 iterations
        start = time.time()
        for i in range(10000):
            f(input_tvm, filter_tvm, output_tvm)
        ctx.sync()
        elapsed_time = time.time() - start   
        print("Time cost of 10000 iterations = %g sec" % elapsed_time)
        # correctness with scipy's convolve2d       
        output_scipy = np.zeros((out_batch, out_channel, out_height, out_width), dtype=Output.dtype)
        for i in range(out_batch):
            for j in range(out_channel):
                output_scipy[i,j,:,:] = signal.convolve2d(input_np[i, j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2), mode='valid')        
        np.testing.assert_allclose(output_tvm.asnumpy(), output_scipy, rtol=1e-5)
        
    with tvm.build_config(auto_unroll_max_step=32,
                          auto_unroll_min_depth=0,
                          unroll_explicit=True,
                          detect_global_barrier=False):
        check_device("cuda")
    print "success"

if __name__ == "__main__":
    test_depthwise_conv()

