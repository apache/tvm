import os
import tvm
import time
import numpy as np
from scipy import signal
from topi.nn.util import *
from tvm.contrib import nvcc_compiler

from topi.ewise import relu
from topi.nn import scale_shift, depthwise_conv2d
from topi.cuda.depthwise_conv2d_map import schedule_depthwise_conv2d_map

TASK = "depthwise_conv2d_map"
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

def test_depthwise_conv2d_map():
    """You may test different settings."""
    batch = 2
    in_channel = 256
    in_height = 32
    in_width = 32

    filter_channel = in_channel
    channel_multiplier = 2
    filter_height = 5
    filter_width = 5

    stride_h = 2
    stride_w = 2

    padding = 'SAME' # or 'VALID'

    # Placeholder
    Input = tvm.placeholder((batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Stride = tvm.nd.array(np.array([stride_h, stride_w]))
    Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')
    # Declare
    DepthwiseConv2d = depthwise_conv2d(Input, Filter, Stride, padding)
    ScaleShift = scale_shift(DepthwiseConv2d, Scale, Shift)
    Relu = relu(ScaleShift)
    # Schedule
    s1 = schedule_depthwise_conv2d_map(DepthwiseConv2d.op)
    s2 = schedule_depthwise_conv2d_map(ScaleShift.op)
    s3 = schedule_depthwise_conv2d_map(Relu.op)

    def depthwise_conv2d_map_scipy(input_np, filter_np, scale_np, shift_np):
        out_shape = get_const_tuple(DepthwiseConv2d.shape)
        out_channel = out_shape[1]
        out_height = out_shape[2]
        out_width = out_shape[3]
        depthwise_conv2d_scipy = np.zeros((batch, out_channel, out_height, out_width), dtype=DepthwiseConv2d.dtype)
        scale_shift_scipy = np.zeros((batch, out_channel, out_height, out_width), dtype=ScaleShift.dtype)
        relu_scipy = np.zeros((batch, out_channel, out_height, out_width), dtype=Relu.dtype)
        if padding == 'SAME':
            pad_top_tvm = np.int(np.ceil(float(np.max((out_height - 1) * stride_h + filter_height - in_height, 0)) / 2))
            pad_left_tvm = np.int(np.ceil(float(np.max((out_width - 1) * stride_w + filter_width - in_width, 0)) / 2))
            pad_top_scipy = np.int(np.ceil(float(filter_height - 1) / 2))
            pad_left_scipy = np.int(np.ceil(float(filter_width - 1) / 2))
            index_h = pad_top_scipy - pad_top_tvm
            index_w = pad_left_scipy - pad_left_tvm
            for i in range(batch):
                for j in range(out_channel):
                    depthwise_conv2d_scipy[i,j,:,:] = signal.convolve2d(input_np[i,j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2),
                        mode='same')[index_h:in_height:stride_h, index_w:in_width:stride_w]
        if padding == 'VALID':
            for i in range(batch):
                for j in range(out_channel):
                    depthwise_conv2d_scipy[i,j,:,:] = signal.convolve2d(input_np[i,j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2),
                        mode='valid')[0:(in_height - filter_height + 1):stride_h, 0:(in_width - filter_height + 1):stride_w]
        for c in range(out_channel):
            scale_shift_scipy[:,c,:,:] = depthwise_conv2d_scipy[:,c,:,:] * scale_np[c] + shift_np[c]
        relu_scipy[:,:,:,:] = np.maximum(scale_shift_scipy[:,:,:,:], 0)
        return depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # Build the kernel
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device)
        f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)
        # Prepare data
        input_np = np.random.uniform(size=get_const_tuple(Input.shape)).astype(Input.dtype)
        filter_np = np.random.uniform(size=get_const_tuple(Filter.shape)).astype(Filter.dtype)
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        scale_np = np.random.uniform(size=(in_channel * channel_multiplier)).astype(Scale.dtype)
        shift_np = np.random.uniform(size=(in_channel * channel_multiplier)).astype(Shift.dtype)
        scale_tvm = tvm.nd.array(scale_np, ctx)
        shift_tvm = tvm.nd.array(shift_np, ctx)
        depthwise_conv2d_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), ctx)
        scale_shift_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), ctx)
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), ctx)
        # Launch kernel 1 (depthwise_conv2d)
        f1(input_tvm, filter_tvm, depthwise_conv2d_tvm)
        ctx.sync()
        start = time.time()
        for i in range(10000):
            f1(input_tvm, filter_tvm, depthwise_conv2d_tvm)
        ctx.sync()
        elapsed_time_1 = time.time() - start
        # Launch kernel 2 (depthwise_conv2d + scale_shift)
        f2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm)
        ctx.sync()
        start = time.time()
        for i in range(10000):
            f2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm)
        ctx.sync()
        elapsed_time_2 = time.time() - start
        # Launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        f3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm)
        ctx.sync()
        start = time.time()
        for i in range(10000):
            f3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm)
        ctx.sync()
        elapsed_time_3 = time.time() - start
        print("Input shape = " + str(get_const_tuple(Input.shape)))
        print("Filter shape = " + str(get_const_tuple(Filter.shape)))
        print("Stride = (%d, %d)" % (stride_h, stride_w))
        print("padding = %s\n" % padding)
        print("Output shape = " + str(get_const_tuple(DepthwiseConv2d.shape)))
        print("time cost of 10000 iterations (depthwise_conv2d) = %g sec" % elapsed_time_1)
        print("time cost of 10000 iterations (depthwise_conv2d + scale_shift) = %g sec" % elapsed_time_2)
        print("time cost of 10000 iterations (depthwise_conv2d + scale_shift + relu) = %g sec" % elapsed_time_3)
        depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy = depthwise_conv2d_map_scipy(input_np, filter_np, scale_np, shift_np)
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5)
        np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)
        print "success"

    unroll_explicit = (get_const_tuple(DepthwiseConv2d.shape)[2] % 8 == 0)
    with tvm.build_config(auto_unroll_max_step=32,
                          auto_unroll_min_depth=0,
                          unroll_explicit=unroll_explicit,
                          detect_global_barrier=False,
                          restricted_func=True):
        check_device("cuda")

if __name__ == "__main__":
    test_depthwise_conv2d_map()
