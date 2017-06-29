"""Depthwise convolution example.

Input: 4-D tensor with shape [batch, in_channel, in_height, in_width]
Filter: 4-D tensor with shape [in_channel, channel_multiplier, filter_height, filter_width]
Stride: 1-D of size 2 
Padding: A string, either 'VALID' or 'SAME'

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

def depthwise_conv(Input, Filter, Stride, padding):  
    for i in range(4):
        assert(isinstance(Input.shape[i], tvm.expr.IntImm))
        assert(isinstance(Filter.shape[i], tvm.expr.IntImm))    
    in_batch = Input.shape[0].value
    in_channel = Input.shape[1].value
    in_height = Input.shape[2].value
    in_width = Input.shape[3].value
    filter_channel = Filter.shape[0].value
    channel_multiplier = Filter.shape[1].value
    filter_height = Filter.shape[2].value
    filter_width = Filter.shape[3].value   
    if not in_channel==filter_channel:
        print("Input channel and filter channel doesn't match!")
        return
    if not in_height * in_width + filter_height * filter_width < 12000:
        print("Can't load input and filter into shared memory (exceeds 48K size limit)!")
        return
    stride_h = Stride.asnumpy()[0]
    stride_w = Stride.asnumpy()[1]

    if padding == 'VALID':
        # calculate output size
        out_batch = in_batch
        out_channel = in_channel * channel_multiplier
        out_height = (in_height - filter_height) / stride_h + 1
        out_width = (in_width - filter_width) / stride_w + 1
        # 2-D sum reduction
        di = tvm.reduce_axis((0, filter_height), name='di')
        dj = tvm.reduce_axis((0, filter_width), name='dj')
        Output = tvm.compute(
            (out_batch, out_channel, out_height, out_width), 
            lambda b, c, i, j: tvm.sum(Input[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] * Filter[c/channel_multiplier, c%channel_multiplier, di, dj], axis=[di, dj]),
            name='Output')

    if padding == 'SAME':
        # calculate output size
        out_batch = in_batch
        out_channel = in_channel * channel_multiplier
        out_height = np.int(np.ceil(float(in_height) / float(stride_h)))
        out_width  = np.int(np.ceil(float(in_width) / float(stride_w)))
        # padding stage
        pad_along_height = np.int(np.max((out_height - 1) * stride_h + filter_height - in_height, 0))
        pad_along_width = np.int(np.max((out_width - 1) * stride_w + filter_width - in_width, 0))
        height_after_pad = in_height + pad_along_height 
        width_after_pad = in_width + pad_along_width
        pad_top = np.int(np.ceil(float(pad_along_height) / 2))
        pad_left = np.int(np.ceil(float(pad_along_width) / 2))
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        PaddedInput = tvm.compute(
            (in_batch, in_channel, height_after_pad, width_after_pad),
            lambda b, c, i, j: tvm.select(tvm.all(i >= pad_top, i - pad_top < in_height, j >= pad_left, j - pad_left < in_width), 
                Input[b, c, i - pad_top, j - pad_left], Input[b, c, i - pad_top, j - pad_left]*0),
            name="PaddedInput")
        # 2-D sum reduction
        di = tvm.reduce_axis((0, filter_height), name='di')
        dj = tvm.reduce_axis((0, filter_width), name='dj')
        Output = tvm.compute(
            (out_batch, out_channel, out_height, out_width), 
            lambda b, c, i, j: tvm.sum(PaddedInput[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] * Filter[c/channel_multiplier, c%channel_multiplier, di, dj], axis=[di, dj]),
            name='Output')

    # schedule
    s = tvm.create_schedule(Output.op)  
    if padding == 'VALID':
        IS = s.cache_read(Input, "shared", [Output])
    if padding == 'SAME':
        IS = s.cache_read(PaddedInput, "shared", [Output])
        s[PaddedInput].compute_inline()
    FS = s.cache_read(Filter, "shared", [Output]) 
    IL = s.cache_read(IS, "local", [Output])
    FL = s.cache_read(FS, "local", [Output])
    
    num_thread = 8
    num_vthread_x = 1
    num_vthread_y = 1
    if out_height % 32 == 0:
        num_vthread_x = out_height / 32
    elif out_height % 16 == 0:
        num_vthread_x = out_height / 16
    if out_width % 32 == 0:
        num_vthread_y = out_width / 32
    elif out_width % 16 == 0:
        num_vthread_y = out_width / 16     
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
    thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")

    s[Output].bind(Output.op.axis[0], block_y)
    bx, xi = s[Output].split(Output.op.axis[1], factor=channel_multiplier)
    s[Output].bind(bx, block_x) 
    s[Output].reorder(Output.op.axis[2], Output.op.axis[3], xi)  
    tvx, xi = s[Output].split(Output.op.axis[2], nparts=num_vthread_x)
    tx, xi = s[Output].split(xi, nparts=num_thread)
    tvy, yi = s[Output].split(Output.op.axis[3], nparts=num_vthread_y)
    ty, yi = s[Output].split(yi, nparts=num_thread)
    s[Output].bind(tvx, thread_vx)
    s[Output].bind(tvy, thread_vy)
    s[Output].bind(tx, thread_x)
    s[Output].bind(ty, thread_y)
    s[Output].reorder(tvx, tvy, tx, ty, xi, yi)
 
    s[IL].compute_at(s[Output], ty)
    s[FL].compute_at(s[Output], ty)

    # schedule for input's shared memory load
    s[IS].compute_at(s[Output], bx)   
    tx, xi = s[IS].split(IS.op.axis[2], nparts=num_thread)
    ty, yi = s[IS].split(IS.op.axis[3], nparts=num_thread)
    s[IS].bind(tx, thread_x)
    s[IS].bind(ty, thread_y)
    # schedule for filter's shared memory load
    s[FS].compute_at(s[Output], bx)
    s[FS].reorder(FS.op.axis[2], FS.op.axis[3], FS.op.axis[1])  
    tx, xi = s[FS].split(FS.op.axis[2], nparts=num_thread)
    ty, yi = s[FS].split(FS.op.axis[3], nparts=num_thread)
    s[FS].bind(tx, thread_x)
    s[FS].bind(ty, thread_y)

    return s, Output

def test_depthwise_conv():
    in_batch = 2
    in_channel = 256
    in_height = 32
    in_width = 32

    filter_channel = in_channel
    channel_multiplier = 2
    filter_height = 5
    filter_width = 5

    stride_h = 2
    stride_w = 2

    Input = tvm.placeholder((in_batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Stride = tvm.nd.array(np.array([stride_h, stride_w]))
    padding = 'SAME' # or 'VALID' 

    schedule, Output = depthwise_conv(Input, Filter, Stride, padding)

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
        for i in range(4):
            assert(isinstance(Output.shape[i], tvm.expr.IntImm))
        out_batch = Output.shape[0].value
        out_channel = Output.shape[1].value
        out_height = Output.shape[2].value
        out_width = Output.shape[3].value
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
        if padding == 'SAME':
            pad_top_tvm = np.int(np.ceil(float(np.max((out_height - 1) * stride_h + filter_height - in_height, 0)) / 2))
            pad_left_tvm = np.int(np.ceil(float(np.max((out_width - 1) * stride_w + filter_width - in_width, 0)) / 2))
            pad_top_scipy = np.int(np.ceil(float(filter_height - 1) / 2))
            pad_left_scipy = np.int(np.ceil(float(filter_width - 1) / 2))
            index_h = pad_top_scipy - pad_top_tvm
            index_w = pad_left_scipy - pad_left_tvm
            for i in range(out_batch):
                for j in range(out_channel):
                    output_scipy[i,j,:,:] = signal.convolve2d(input_np[i, j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2), 
                        mode='same')[index_h:in_height:stride_h, index_w:in_width:stride_w]   
        if padding == 'VALID':
            for i in range(out_batch):
                for j in range(out_channel):
                    output_scipy[i,j,:,:] = signal.convolve2d(input_np[i, j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2), 
                        mode='valid')[0:(in_height - filter_height + 1):stride_h, 0:(in_width - filter_height + 1):stride_w]  
        np.testing.assert_allclose(output_tvm.asnumpy(), output_scipy, rtol=1e-5)
        print "success"
        
    with tvm.build_config(auto_unroll_max_step=32,
                          auto_unroll_min_depth=0,
                          unroll_explicit=True,
                          detect_global_barrier=False):
        check_device("cuda")

if __name__ == "__main__":
    test_depthwise_conv()

