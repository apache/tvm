
import tvm
import numpy as np
from topi.util import get_const_tuple
from depthwise_conv import depthwise_conv
from scipy import signal
import time

def test_depthwise_conv():
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
    bn_relu_fusion = True # or 'False'

    Input = tvm.placeholder((batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Stride = tvm.nd.array(np.array([stride_h, stride_w]))
    if bn_relu_fusion == True:
        BNparams = tvm.placeholder((in_channel * channel_multiplier, 2), name='BNparams')
    else:
        BNparams = None
    
    schedule, Output = depthwise_conv(Input, Filter, Stride, padding, bn_relu_fusion=bn_relu_fusion, BNparams=BNparams)

    def depthwise_conv_scipy(input_np, filter_np, bn_params_np):
        out_shape = get_const_tuple(Output.shape)
        out_channel = out_shape[1]
        out_height = out_shape[2]
        out_width = out_shape[3]
        output_scipy = np.zeros((batch, out_channel, out_height, out_width), dtype=Output.dtype)
        if padding == 'SAME':
            pad_top_tvm = np.int(np.ceil(float(np.max((out_height - 1) * stride_h + filter_height - in_height, 0)) / 2))
            pad_left_tvm = np.int(np.ceil(float(np.max((out_width - 1) * stride_w + filter_width - in_width, 0)) / 2))
            pad_top_scipy = np.int(np.ceil(float(filter_height - 1) / 2))
            pad_left_scipy = np.int(np.ceil(float(filter_width - 1) / 2))
            index_h = pad_top_scipy - pad_top_tvm
            index_w = pad_left_scipy - pad_left_tvm
            for i in range(batch):
                for j in range(out_channel):
                    output_scipy[i,j,:,:] = signal.convolve2d(input_np[i,j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2), 
                        mode='same')[index_h:in_height:stride_h, index_w:in_width:stride_w]   
        if padding == 'VALID':
            for i in range(batch):
                for j in range(out_channel):
                    output_scipy[i,j,:,:] = signal.convolve2d(input_np[i,j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2), 
                        mode='valid')[0:(in_height - filter_height + 1):stride_h, 0:(in_width - filter_height + 1):stride_w] 
        if bn_relu_fusion == True:
            for c in range(out_channel):
                output_scipy[:,c,:,:] = np.maximum((output_scipy[:,c,:,:] * bn_params_np[c,0] + bn_params_np[c,1]), 0)
        return output_scipy

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # Build the kernel
        if bn_relu_fusion == True:
            f = tvm.build(schedule, [Input, Filter, BNparams, Output], device)
        else: 
            f = tvm.build(schedule, [Input, Filter, Output], device)
        # Prepare data
        input_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(Input.dtype)
        filter_np = np.random.uniform(size=(in_channel, channel_multiplier, filter_height, filter_width)).astype(Filter.dtype)
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        out_shape = get_const_tuple(Output.shape)
        out_channel = out_shape[1]
        out_height = out_shape[2]
        out_width = out_shape[3]
        output_tvm = tvm.nd.array(np.zeros((batch, out_channel, out_height, out_width), dtype=Output.dtype), ctx)
        bn_params_np = None
        if bn_relu_fusion == True:
            data_mean = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
            data_var = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
            gamma = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
            beta = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
            scale = gamma / np.sqrt(data_var + 1e-5)
            shift = beta - data_mean * gamma / np.sqrt(data_var + 1e-5)
            bn_params_np = np.vstack((scale, shift)).T
            bn_params_tvm = tvm.nd.array(bn_params_np, ctx)   
        # Launch the kernel with fusion 
        if bn_relu_fusion == True:
            # Skip first pass as it is compilation
            f(input_tvm, filter_tvm, bn_params_tvm, output_tvm)
            ctx.sync()
            # Measure time cost of 10000 iterations
            start = time.time()
            for i in range(10000):
                f(input_tvm, filter_tvm, bn_params_tvm, output_tvm)
            ctx.sync()
            elapsed_time = time.time() - start 
        # Launch the kernel without fusion 
        if bn_relu_fusion == False:
            f(input_tvm, filter_tvm, output_tvm)
            ctx.sync()
            start = time.time()
            for i in range(10000):
                f(input_tvm, filter_tvm, output_tvm)
            ctx.sync()
            elapsed_time = time.time() - start 
        print("Input shape = [%d, %d, %d, %d]" % (batch, in_channel, in_height, in_width))
        print("Filter shape = [%d, %d, %d, %d]" % (filter_channel, channel_multiplier, filter_height, filter_width))
        print("Stride = [%d, %d]" % (stride_h, stride_w))
        print("padding = %s" % padding)
        print("bn_relu_fusion = %s\n" % bn_relu_fusion) 
        print("Output shape = [%d, %d, %d, %d]" % (batch, out_channel, out_height, out_width))
        print("time cost of 10000 iterations = %g sec" % elapsed_time)
        output_scipy = depthwise_conv_scipy(input_np, filter_np, bn_params_np=bn_params_np)
        np.testing.assert_allclose(output_tvm.asnumpy(), output_scipy, atol=1e-5, rtol=1e-5)
        print "success"
        
    with tvm.build_config(auto_unroll_max_step=32,
                          auto_unroll_min_depth=0,
                          unroll_explicit=True,
                          detect_global_barrier=False):
        check_device("cuda")

if __name__ == "__main__":
    test_depthwise_conv()
