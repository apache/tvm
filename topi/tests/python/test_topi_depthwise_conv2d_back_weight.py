import tvm
import topi
import topi.testing
import numpy as np
from tvm.contrib.pickle_memoize import memoize
from scipy import signal
from topi.util import get_const_tuple
from topi.nn.util import get_pad_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_backward_weight_nhwc


def verify_depthwise_conv2d_back_weight(batch, in_channel, in_h, channel_multiplier, filter_h, stride_h, padding_h):
    in_w = in_h
    filter_channel = in_channel
    filter_w = filter_h
    stride_w = stride_h
    padding_w = padding_h

    out_h = np.int((in_h+2*padding_h-filter_h)/stride_h+1)
    out_w = np.int((in_w+2*padding_w-filter_w)/stride_w+1)
    out_channel = in_channel * channel_multiplier

    oshape = [batch, out_h, out_w, out_channel]
    fshape = [filter_h, filter_w, in_channel, channel_multiplier]

    # placeholder
    Out_grad = tvm.placeholder(oshape, name='Out_grad')
    Input = tvm.placeholder((batch, in_h, in_w, in_channel), name='In_grad')
    # declare
    Weight_grad = topi.nn.depthwise_conv2d_backward_weight_nhwc(Input, Out_grad, oshape, fshape,
        stride=[stride_h, stride_w], padding=[padding_h, padding_w])
    # schedule
    schedule = schedule_depthwise_conv2d_backward_weight_nhwc(Weight_grad)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        # build the kernel
        f = tvm.build(schedule, [Input, Out_grad, Weight_grad], device)
        # prepare pod type for test data closure
        dtype = Out_grad.dtype
        out_grad_shape = get_const_tuple(Out_grad.shape)
        in_shape = get_const_tuple(Input.shape)

        # use memoize to pickle the test data for next time use
        @memoize("topi.tests.test_topi_depthwise_conv2d_backward_weight.nhwc")
        def get_ref_data():
            out_grad_np = np.random.uniform(size=out_grad_shape).astype(dtype)
            input_np = np.random.uniform(size=in_shape).astype(dtype)
            dilated_out_grad_np = topi.testing.dilate_python(out_grad_np, [1, stride_h, stride_w, 1])

            pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple([padding_h, padding_w], (filter_h, filter_w))
            padded_input_np = np.zeros((batch, in_h+pad_top+pad_bottom, in_w+pad_left+pad_right, in_channel))
            padded_input_np[:, pad_top:in_h+pad_top, pad_left:in_w+pad_left, :] = input_np

            weight_grad_np = np.zeros((filter_h, filter_w, in_channel, channel_multiplier))
            for c in range(in_channel):
                for m in range(channel_multiplier):
                    for b in range(batch):
                        weight_grad_np[:, :, c, m] += signal.convolve2d(padded_input_np[b, :, :, c], \
                            np.rot90(dilated_out_grad_np[b, :, :, c*channel_multiplier+m%channel_multiplier], 2), \
                            mode='valid')[0:filter_h, 0:filter_w]
            return (out_grad_np, input_np, weight_grad_np)

        (out_grad_np, input_np, weight_grad_np) = get_ref_data()

        out_grad_tvm = tvm.nd.array(out_grad_np, ctx)
        input_tvm = tvm.nd.array(input_np, ctx)
        weight_grad_tvm = tvm.nd.array(np.zeros(shape=fshape, dtype=dtype), ctx)
        # launch the kernel
        timer = f.time_evaluator(f.entry_name, ctx, number=1)
        tcost = timer(input_tvm, out_grad_tvm, weight_grad_tvm).mean
        np.testing.assert_allclose(weight_grad_np, weight_grad_tvm.asnumpy(), rtol=1e-4)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")
    check_device("rocm")
    check_device("vulkan")

def test_topi_depthwise_conv2d_backward_weight_nhwc():
    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 3, 1, 1)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 2, 3, 1, 1)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 5, 1, 2)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 2, 5, 1, 2)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 3, 2, 1)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 2, 3, 2, 1)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 5, 2, 2)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 2, 5, 2, 2)

    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 3, 1, 0)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 2, 3, 1, 0)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 5, 1, 0)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 2, 5, 1, 0)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 3, 2, 0)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 2, 3, 2, 0)
    verify_depthwise_conv2d_back_weight(16, 256, 56, 1, 5, 2, 0)
    verify_depthwise_conv2d_back_weight(15, 256, 56, 2, 5, 2, 0)

if __name__ == "__main__":
    test_topi_depthwise_conv2d_backward_weight_nhwc()
