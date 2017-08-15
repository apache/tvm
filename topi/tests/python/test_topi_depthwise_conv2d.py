import tvm
import topi
import numpy as np
from scipy import signal
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d

def depthwise_conv2d_with_workload(batch, in_channel, in_height, channel_multiplier, filter_height, stride_h, padding):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_w = stride_h
    # placeholder
    Input = tvm.placeholder((batch, in_channel, in_height, in_width), name='Input')
    Filter = tvm.placeholder((filter_channel, channel_multiplier, filter_height, filter_width), name='Filter')
    Stride = [stride_h, stride_w]
    Scale = tvm.placeholder((in_channel * channel_multiplier,), name='Scale')
    Shift = tvm.placeholder((in_channel * channel_multiplier,), name='Shift')
    # declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d(Input, Filter, Stride, padding)
    ScaleShift = topi.nn.scale_shift(DepthwiseConv2d, Scale, Shift)
    Relu = topi.nn.relu(ScaleShift)
    # schedule
    s1 = schedule_depthwise_conv2d(DepthwiseConv2d)
    s2 = schedule_depthwise_conv2d(ScaleShift)
    s3 = schedule_depthwise_conv2d(Relu)

    input_np = np.random.uniform(size=get_const_tuple(Input.shape)).astype(Input.dtype)
    filter_np = np.random.uniform(size=get_const_tuple(Filter.shape)).astype(Filter.dtype)
    scale_np = np.random.uniform(size=get_const_tuple(Scale.shape)).astype(Scale.dtype)
    shift_np = np.random.uniform(size=get_const_tuple(Shift.shape)).astype(Shift.dtype)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.context(device, 0)
        # build the kernels
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device)
        f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)
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
        # correctness with scipy
        depthwise_conv2d_scipy = topi.testing.depthwise_conv2d_python(input_np, filter_np, stride=[stride_h, stride_w], padding=padding)
        scale_shift_scipy = np.zeros(shape=get_const_tuple(ScaleShift.shape))
        for c in range(in_channel * channel_multiplier):
            scale_shift_scipy[:,c,:,:] = depthwise_conv2d_scipy[:,c,:,:] * scale_np[c] + shift_np[c]
        relu_scipy = np.maximum(scale_shift_scipy, 0)
        np.testing.assert_allclose(depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5)
        np.testing.assert_allclose(scale_shift_tvm.asnumpy(), scale_shift_scipy, rtol=1e-5)
        np.testing.assert_allclose(relu_tvm.asnumpy(), relu_scipy, rtol=1e-5)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")


def test_depthwise_conv2d():
    depthwise_conv2d_with_workload(1, 728, 64, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload(1, 728, 32, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload(4, 256, 64, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload(4, 256, 32, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload(1, 728, 64, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload(1, 728, 32, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload(4, 256, 64, 2, 5, 2, "VALID")
    depthwise_conv2d_with_workload(4, 256, 32, 2, 5, 2, "VALID")


if __name__ == "__main__":
    test_depthwise_conv2d()
