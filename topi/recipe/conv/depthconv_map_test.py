
import tvm
import numpy as np
from topi.util import get_const_tuple
from depthconv_map import *
from scipy import signal
import time

def test_depthconv_map():
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
    BNparams = tvm.placeholder((in_channel * channel_multiplier, 2), name='BNparams')
    # Declare
    DepthConv = depthconv(Input, Filter, Stride, padding)
    BatchNorm = batchnorm(DepthConv, BNparams)
    Relu = relu(BatchNorm)
    # Schedule
    s1 = schedule_depthconv_map(DepthConv.op)
    s2 = schedule_depthconv_map(BatchNorm.op)
    s3 = schedule_depthconv_map(Relu.op)

    def depthconv_map_scipy(input_np, filter_np, bn_params_np):
        out_shape = get_const_tuple(DepthConv.shape)
        out_channel = out_shape[1]
        out_height = out_shape[2]
        out_width = out_shape[3]
        depthconv_scipy = np.zeros((batch, out_channel, out_height, out_width), dtype=DepthConv.dtype)
        batchnorm_scipy = np.zeros((batch, out_channel, out_height, out_width), dtype=BatchNorm.dtype)
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
                    depthconv_scipy[i,j,:,:] = signal.convolve2d(input_np[i,j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2), 
                        mode='same')[index_h:in_height:stride_h, index_w:in_width:stride_w]   
        if padding == 'VALID':
            for i in range(batch):
                for j in range(out_channel):
                    depthconv_scipy[i,j,:,:] = signal.convolve2d(input_np[i,j/channel_multiplier,:,:], np.rot90(filter_np[j/channel_multiplier,j%channel_multiplier,:,:], 2), 
                        mode='valid')[0:(in_height - filter_height + 1):stride_h, 0:(in_width - filter_height + 1):stride_w] 
        for c in range(out_channel):
            batchnorm_scipy[:,c,:,:] = depthconv_scipy[:,c,:,:] * bn_params_np[c,0] + bn_params_np[c,1]
        relu_scipy[:,:,:,:] = np.maximum(batchnorm_scipy[:,:,:,:], 0)
        return depthconv_scipy, batchnorm_scipy, relu_scipy

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # Build the kernel
        f1 = tvm.build(s1, [Input, Filter, DepthConv], device)
        f2 = tvm.build(s2, [Input, Filter, BNparams, BatchNorm], device)
        f3 = tvm.build(s3, [Input, Filter, BNparams, Relu], device)
        # Prepare data
        input_np = np.random.uniform(size=get_const_tuple(Input.shape)).astype(Input.dtype)
        filter_np = np.random.uniform(size=get_const_tuple(Filter.shape)).astype(Filter.dtype)
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        data_mean = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
        data_var = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
        gamma = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
        beta = np.random.uniform(size=(in_channel * channel_multiplier)).astype(BNparams.dtype)
        scale = gamma / np.sqrt(data_var + 1e-5)
        shift = beta - data_mean * gamma / np.sqrt(data_var + 1e-5)
        bn_params_np = np.vstack((scale, shift)).T
        bn_params_tvm = tvm.nd.array(bn_params_np, ctx)   
        depthconv_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthConv.shape), dtype=DepthConv.dtype), ctx)
        batchnorm_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(BatchNorm.shape), dtype=BatchNorm.dtype), ctx)
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), ctx)
        # Launch kernel 1 (depthconv)
        f1(input_tvm, filter_tvm, depthconv_tvm)
        ctx.sync()
        start = time.time()
        for i in range(10000):
            f1(input_tvm, filter_tvm, depthconv_tvm)
        ctx.sync()
        elapsed_time_1 = time.time() - start 
        # Launch kernel 2 (depthconv + bn)
        f2(input_tvm, filter_tvm, bn_params_tvm, batchnorm_tvm)
        ctx.sync()
        start = time.time()
        for i in range(10000):
            f2(input_tvm, filter_tvm, bn_params_tvm, batchnorm_tvm)
        ctx.sync()
        elapsed_time_2 = time.time() - start 
        # Launch kernel 3 (depthconv + bn + relu)
        f3(input_tvm, filter_tvm, bn_params_tvm, relu_tvm)
        ctx.sync()
        start = time.time()
        for i in range(10000):
            f3(input_tvm, filter_tvm, bn_params_tvm, relu_tvm)
        ctx.sync()
        elapsed_time_3 = time.time() - start 
        print("Input shape = " + str(get_const_tuple(Input.shape)))
        print("Filter shape = " + str(get_const_tuple(Filter.shape)))
        print("Stride = (%d, %d)" % (stride_h, stride_w))
        print("padding = %s\n" % padding)
        print("Output shape = " + str(get_const_tuple(DepthConv.shape)))
        print("time cost of 10000 iterations (depthconv) = %g sec" % elapsed_time_1)
        print("time cost of 10000 iterations (depthconv + bn) = %g sec" % elapsed_time_2)
        print("time cost of 10000 iterations (depthconv + bn + relu) = %g sec" % elapsed_time_3)
        depthconv_scipy, batchnorm_scipy, relu_scipy = depthconv_map_scipy(input_np, filter_np, bn_params_np)
        np.testing.assert_allclose(depthconv_tvm.asnumpy(), depthconv_scipy, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(batchnorm_tvm.asnumpy(), batchnorm_scipy, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, atol=1e-5, rtol=1e-5)
        print "success"
        
    with tvm.build_config(auto_unroll_max_step=32,
                          auto_unroll_min_depth=0,
                          unroll_explicit=True,
                          detect_global_barrier=False,
                          restricted_func=True):
        check_device("cuda")

if __name__ == "__main__":
    test_depthconv_map()
