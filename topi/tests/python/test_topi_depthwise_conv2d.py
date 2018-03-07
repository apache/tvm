import tvm
import topi
import topi.testing
import numpy as np
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_nhwc


def depthwise_conv2d_with_workload_nchw(batch, in_channel, in_height, channel_multiplier, filter_height, stride, padding):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    # placeholder
    Input = tvm.placeholder((batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')
    # declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nchw(Input, Filter, stride=stride, padding=padding)
    ScaleShift = topi.nn.scale_shift_nchw(DepthwiseConv2d, Scale, Shift)
    Relu = topi.nn.relu(ScaleShift)


    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            # schedule
            s1 = topi.generic.schedule_depthwise_conv2d_nchw(DepthwiseConv2d)
            s2 = topi.generic.schedule_depthwise_conv2d_nchw(ScaleShift)
            s3 = topi.generic.schedule_depthwise_conv2d_nchw(Relu)
        # build the kernels
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device)
        f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)

        # Prepare pod type for test data closure
        dtype = Input.dtype
        input_shape = get_const_tuple(Input.shape)
        filter_shape = get_const_tuple(Filter.shape)
        scale_shape = get_const_tuple(Scale.shape)
        shift_shape = get_const_tuple(Shift.shape)
        scale_shift_shape = get_const_tuple(ScaleShift.shape)

        # Use memoize, pickle the test data for next time use.
        @memoize("topi.tests.test_topi_depthwise_conv2d.nchw")
        def get_ref_data():
            input_np = np.random.uniform(size=input_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            scale_np = np.random.uniform(size=scale_shape).astype(dtype)
            shift_np = np.random.uniform(size=shift_shape).astype(dtype)
            # correctness with scipy
            depthwise_conv2d_scipy = topi.testing.depthwise_conv2d_python_nchw(
                input_np, filter_np, stride=stride, padding=padding)
            scale_shift_scipy = np.zeros(shape=scale_shift_shape)
            for c in range(in_channel * channel_multiplier):
                scale_shift_scipy[:,c,:,:] = depthwise_conv2d_scipy[:,c,:,:] * scale_np[c] + shift_np[c]
                relu_scipy = np.maximum(scale_shift_scipy, 0)
            return (input_np, filter_np, scale_np, shift_np,
                    depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy)
        # Get the test data
        (input_np, filter_np, scale_np, shift_np,
         depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy) = get_ref_data()

        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        scale_tvm = tvm.nd.array(scale_np, ctx)
        shift_tvm = tvm.nd.array(shift_np, ctx)
        depthwise_conv2d_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), ctx)
        scale_shift_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), ctx)
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), ctx)
        # launch kernel 1 (depthwise_conv2d)
        timer_1 = f1.time_evaluator(f1.entry_name, ctx, number=1)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # launch kernel 2 (depthwise_conv2d + scale_shift)
        timer_2 = f2.time_evaluator(f2.entry_name, ctx, number=1)
        tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        timer_3 = f3.time_evaluator(f3.entry_name, ctx, number=1)
        tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5)
        np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")
    check_device("rocm")
    check_device("vulkan")


def depthwise_conv2d_with_workload_nhwc(batch, in_channel, in_height, channel_multiplier, filter_height, stride_h, padding):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_w = stride_h
    # placeholder
    Input = tvm.placeholder((batch, in_height, in_width, in_channel), name='Input')
    Filter = tvm.placeholder((filter_height, filter_width,filter_channel, channel_multiplier), name='Filter')
    Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')
    # declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nhwc(Input, Filter, stride=[stride_h, stride_w], padding=padding)
    ScaleShift = topi.nn.scale_shift_nhwc(DepthwiseConv2d, Scale, Shift)
    Relu = topi.nn.relu(ScaleShift)
    # schedule

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            s1 = topi.generic.schedule_depthwise_conv2d_nhwc(DepthwiseConv2d)
            s2 = topi.generic.schedule_depthwise_conv2d_nhwc(ScaleShift)
            s3 = topi.generic.schedule_depthwise_conv2d_nhwc(Relu)
        # build the kernels
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device)
        f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)

        # Prepare pod type for test data closure
        dtype = Input.dtype
        input_shape = get_const_tuple(Input.shape)
        filter_shape = get_const_tuple(Filter.shape)
        scale_shape = get_const_tuple(Scale.shape)
        shift_shape = get_const_tuple(Shift.shape)
        scale_shift_shape = get_const_tuple(ScaleShift.shape)

        # Use memoize, pickle the test data for next time use.
        @memoize("topi.tests.test_topi_depthwise_conv2d.nhwc")
        def get_ref_data():
            input_np = np.random.uniform(size=input_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            scale_np = np.random.uniform(size=scale_shape).astype(dtype)
            shift_np = np.random.uniform(size=shift_shape).astype(dtype)
            # correctness with scipy
            depthwise_conv2d_scipy = topi.testing.depthwise_conv2d_python_nhwc(
                input_np, filter_np, stride=[stride_h, stride_w], padding=padding)
            scale_shift_scipy = np.zeros(shape=scale_shift_shape)
            for c in range(in_channel * channel_multiplier):
                scale_shift_scipy[:,:,:,c] = depthwise_conv2d_scipy[:,:,:,c] * scale_np[c] + shift_np[c]
                relu_scipy = np.maximum(scale_shift_scipy, 0)
            return (input_np, filter_np, scale_np, shift_np,
                    depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy)
        # Get the test data
        (input_np, filter_np, scale_np, shift_np,
         depthwise_conv2d_scipy, scale_shift_scipy, relu_scipy) = get_ref_data()

        # prepare data
        input_tvm = tvm.nd.array(input_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)
        scale_tvm = tvm.nd.array(scale_np, ctx)
        shift_tvm = tvm.nd.array(shift_np, ctx)
        depthwise_conv2d_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), ctx)
        scale_shift_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), ctx)
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), ctx)
        # launch kernel 1 (depthwise_conv2d)
        timer_1 = f1.time_evaluator(f1.entry_name, ctx, number=1)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # launch kernel 2 (depthwise_conv2d + scale_shift)
        timer_2 = f2.time_evaluator(f2.entry_name, ctx, number=1)
        tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        timer_3 = f3.time_evaluator(f3.entry_name, ctx, number=1)
        tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        relu_scipy = np.maximum(scale_shift_scipy, 0)
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5)
        np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")
    check_device("rocm")
    check_device("vulkan")

def test_depthwise_conv2d():
    print("testing nchw")
    depthwise_conv2d_with_workload_nchw(1, 728, 64, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nchw(1, 728, 32, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nchw(4, 256, 64, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload_nchw(4, 256, 32, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload_nchw(1, 728, 64, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload_nchw(1, 728, 32, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload_nchw(4, 256, 64, 2, 5, 2, "VALID")
    depthwise_conv2d_with_workload_nchw(4, 256, 32, 2, 5, 2, "VALID")
    print("testing nhwc")
    depthwise_conv2d_with_workload_nhwc(1, 728, 64, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nhwc(1, 728, 32, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nhwc(4, 256, 64, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload_nhwc(4, 256, 32, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload_nhwc(1, 728, 64, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload_nhwc(1, 728, 32, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload_nhwc(4, 256, 64, 2, 5, 2, "VALID")
    depthwise_conv2d_with_workload_nhwc(4, 256, 32, 2, 5, 2, "VALID")

if __name__ == "__main__":
    test_depthwise_conv2d()
