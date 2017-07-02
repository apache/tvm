
import tvm
import numpy as np
from topi.util import get_const_tuple
from cuda_depthwise_conv import depthwise_conv
from scipy import signal
import time

def test_depthwise_conv():
    """You may test different settings"""
    in_batch = 4
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

    Input = tvm.placeholder((in_batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Stride = tvm.nd.array(np.array([stride_h, stride_w]))
    
    schedule, Output = depthwise_conv(Input, Filter, Stride, padding)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        f = tvm.build(schedule, [Input, Filter, Output], device)
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # Launch the kernel
        input_np = np.random.uniform(size=(in_batch, in_channel, in_height, in_width)).astype(Input.dtype)
        filter_np = np.random.uniform(size=(in_channel, channel_multiplier, filter_height, filter_width)).astype(Filter.dtype)
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        out_shape = get_const_tuple(Output.shape)
        out_batch = out_shape[0]
        out_channel = out_shape[1]
        out_height = out_shape[2]
        out_width = out_shape[3]
        output_tvm = tvm.nd.array(np.zeros((out_batch, out_channel, out_height, out_width), dtype=Output.dtype), ctx)
        # Skip first pass as it is compilation
        f(input_tvm, filter_tvm, output_tvm)
        ctx.sync()
        # Measure time cost of 10000 iterations
        start = time.time()
        for i in range(10000):
            f(input_tvm, filter_tvm, output_tvm)
        ctx.sync()
        elapsed_time = time.time() - start   
        print("Input shape = [%d, %d, %d, %d]" % (in_batch, in_channel, in_height, in_width))
        print("Filter shape = [%d, %d, %d, %d]" % (filter_channel, channel_multiplier, filter_height, filter_width))
        print("Stride = [%d, %d]" % (stride_h, stride_w))
        print("padding = %s\n" % padding)
        print("Output shape = [%d, %d, %d, %d]" % (out_batch, out_channel, out_height, out_width))
        print("time cost of 10000 iterations = %g sec" % elapsed_time)
        # Correctness with scipy's convolve2d       
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
