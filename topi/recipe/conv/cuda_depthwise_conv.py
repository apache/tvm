"""Depthwise convolution example, 12x faster than tensorflow's depthwise_conv2d

This is an example for depthwise convolution, 
its counterpart operation in tensorflow is:

```python
import tensorflow as tf
input = tf.Variable(tf.random_uniform([1, 728, 21, 21]))
filter = tf.Variable(tf.random_uniform([3, 3, 728, 1])) 
output = tf.nn.depthwise_conv2d(input, filter, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
```
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

def test_depthwise_conv():
    # graph 
    in_channel = 728
    in_height = 21
    in_width = 21

    filter_channel = in_channel
    filter_height = 3
    filter_width = 3

    out_channel = in_channel
    out_height = 19
    out_width = 19

    ic = tvm.var('ic')
    ic = tvm.convert(in_channel)
    ih = tvm.var('ih')
    ih = tvm.convert(in_height)
    iw = tvm.var('iw')
    iw = tvm.convert(in_width)
    Input = tvm.placeholder((ic, ih, iw), name='Input')

    fc = tvm.var('fc')
    fc = tvm.convert(filter_channel)
    fh = tvm.var('fh')
    fh = tvm.convert(filter_height)
    fw = tvm.var('fw')
    fw = tvm.convert(filter_width)
    Filter = tvm.placeholder((fc, fh, fw), name='Filter')

    oc = tvm.var('oc')
    oc = tvm.convert(out_channel)
    oh = tvm.var('oh')
    oh = tvm.convert(out_height)
    ow = tvm.var('ow')
    ow = tvm.convert(out_width)
    Output = tvm.compute(
        (oc, oh, ow),
        lambda kk, ii, jj: Input[kk, ii, jj]*Filter[kk, 0, 0] + Input[kk, ii, jj+1]*Filter[kk, 0, 1] + Input[kk, ii, jj+2]*Filter[kk, 0, 2] +
                           Input[kk, ii+1, jj]*Filter[kk, 1, 0] + Input[kk, ii+1, jj+1]*Filter[kk, 1, 1] + Input[kk, ii+1, jj+2]*Filter[kk, 1, 2] + 
                           Input[kk, ii+2, jj]*Filter[kk, 2, 0] + Input[kk, ii+2, jj+1]*Filter[kk, 2, 1] + Input[kk, ii+2, jj+2]*Filter[kk, 2, 2],
        name='Output')

    # schedule
    s = tvm.create_schedule(Output.op)
    IS = s.cache_read(Input, "shared", [Output])
    FS = s.cache_read(Filter, "shared", [Output])

    num_thread = 8
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")

    s[Output].bind(Output.op.axis[0], block_x)
    tx, xi = s[Output].split(Output.op.axis[1], nparts=num_thread)
    ty, yi = s[Output].split(Output.op.axis[2], nparts=num_thread)
    s[Output].bind(tx, thread_x)
    s[Output].bind(ty, thread_y)
    s[Output].reorder(tx, ty, xi, yi)

    s[IS].compute_at(s[Output], Output.op.axis[0])
    s[FS].compute_at(s[Output], Output.op.axis[0])

    tx, xi = s[IS].split(IS.op.axis[1], nparts=num_thread)
    ty, yi = s[IS].split(IS.op.axis[2], nparts=num_thread)
    s[IS].bind(tx, thread_x)
    s[IS].bind(ty, thread_y)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        f = tvm.build(s, [Input, Filter, Output], device)
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # launch the kernel
        input_np = np.random.uniform(size=(in_channel, in_height, in_width)).astype(Input.dtype)
        filter_np = np.random.uniform(size=(filter_channel, filter_height, filter_width)).astype(Filter.dtype)
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        output_tvm = tvm.nd.array(np.zeros((out_channel, out_height, out_width), dtype=Output.dtype), ctx)
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
        output_scipy = np.zeros((out_channel, out_height, out_width), dtype=Output.dtype)
        for i in range(out_channel):
            output_scipy[i,:,:] = signal.convolve2d(input_np[i,:,:], np.rot90(filter_np[i,:,:], 2), mode='valid')        
        np.testing.assert_allclose(
            output_tvm.asnumpy(), output_scipy, rtol=1e-5)
        print "success"
        
    with tvm.build_config(auto_unroll_max_step=32,
                          auto_unroll_min_depth=0,
                          unroll_explicit=False):
        check_device("cuda")

if __name__ == "__main__":
    test_depthwise_conv()
