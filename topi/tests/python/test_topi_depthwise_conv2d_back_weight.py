import tvm
import topi
import numpy as np
from tvm.contrib.pickle_memoize import memoize
from scipy import signal
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_backward_weight_nhwc

def depthwise_conv2d_with_workload_nhwc(batch, in_channel, in_height, channel_multiplier, filter_height, stride_h, padding_h):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_w = stride_h
    padding_w = padding_h

    out_height = np.int((in_height+2*padding_h-filter_height)/stride_h+1)
    out_width = np.int((in_width+2*padding_w-filter_width)/stride_w+1)
    out_channel = in_channel * channel_multiplier

    oshape = [batch, out_height, out_width, out_channel]
    fshape = [filter_height, filter_width, in_channel, channel_multiplier]

    # placeholder
    Out_grad = tvm.placeholder(oshape, name='Out_grad')
    Input = tvm.placeholder((batch, in_height, in_width, in_channel), name='In_grad')
    stride = [stride_h, stride_w]
    padding = [padding_h, padding_w]

    # declare
    Weight_grad = topi.nn.depthwise_conv2d_backward_weight_nhwc(Input, Out_grad, oshape, fshape, stride, padding)

    # schedule
    schedule = schedule_depthwise_conv2d_backward_weight_nhwc(Weight_grad)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.context(device, 0)

        # build the kernels
        f = tvm.build(schedule, [Input, Out_grad, Weight_grad], device)

        # Prepare pod type for test data closure
        dtype = Out_grad.dtype
        out_grad_shape = get_const_tuple(Out_grad.shape)
        in_shape = get_const_tuple(Input.shape)
        
        @memoize("topi.tests.test_topi_depthwise_conv2d_backward_weight.nhwc")
        def get_ref_data():
            out_backprop_np = np.random.uniform(size=out_grad_shape).astype(dtype)
            input_np = np.random.uniform(size=in_shape).astype(dtype)

            Dilated_out_grad = topi.testing.dilate_python(out_backprop_np, [1, stride_h, stride_w, 1])
            if padding_h == 0 and padding_w == 0:
                weight_grad_np = np.zeros((filter_height, filter_width, in_channel, channel_multiplier))
                for c in range(in_channel):
                    for m  in range(channel_multiplier):
                        for b in range(batch):
                            weight_grad_np[:, :, c, m] += signal.convolve2d(input_np[b, :, :, c], \
                                np.rot90(Dilated_out_grad[b, :, :, c*channel_multiplier+m%channel_multiplier], 2), \
                                mode='valid')[0:filter_height, 0:filter_width]

            if padding_h > 0 or padding_w > 0:
                weight_grad_np = np.zeros((filter_height, filter_width, in_channel, channel_multiplier))
                index_h = int((input_np.shape[1] - filter_height + (input_np.shape[1]+stride_h)%2)/2)
                index_w = int((input_np.shape[2] - filter_width + (input_np.shape[2]+stride_w)%2)/2)
                for c in range(in_channel):
                    for m  in range(channel_multiplier):
                        for b in range(batch):
                            weight_grad_np[:,:,c,m] += signal.convolve2d(input_np[b, :, :, c], \
                                np.rot90(Dilated_out_grad[b, :, :, c*channel_multiplier+m%channel_multiplier], 2), \
                                mode='same')[index_h:(index_h+filter_height), index_w:(index_w+filter_width)]
            return (out_backprop_np, input_np, weight_grad_np)
            
        (out_backprop_np, input_np, weight_grad_np) = get_ref_data()
        
        # prepare data
        out_backprop_tvm = tvm.nd.array(out_backprop_np, ctx)
        input_tvm = tvm.nd.array(input_np, ctx)
        weight_backprop_tvm = tvm.nd.array(np.zeros((filter_height, filter_width, in_channel, channel_multiplier), dtype=Weight_grad.dtype), ctx)

        # launch kernel (depthwise_conv2d backward nhwc wrt input)
        timer = f.time_evaluator(f.entry_name, ctx, number=1)
        tcost = timer(input_tvm, out_backprop_tvm, weight_backprop_tvm).mean
        
        np.testing.assert_allclose(weight_grad_np, weight_backprop_tvm.asnumpy(), rtol=1e-4)

    check_device("cuda")

def test_depthwise_conv2d():
    print("testing nhwc")
    depthwise_conv2d_with_workload_nhwc(17, 32, 17, 1, 3, 1, 1)
    depthwise_conv2d_with_workload_nhwc(17, 32, 18, 1, 3, 1, 1)
    depthwise_conv2d_with_workload_nhwc(18, 32, 17, 2, 5, 1, 2)
    depthwise_conv2d_with_workload_nhwc(18, 32, 18, 2, 5, 1, 2)
    depthwise_conv2d_with_workload_nhwc(17, 32, 17, 1, 3, 2, 1)
    depthwise_conv2d_with_workload_nhwc(17, 32, 18, 1, 3, 2, 1)
    depthwise_conv2d_with_workload_nhwc(18, 32, 17, 2, 5, 2, 2)
    depthwise_conv2d_with_workload_nhwc(18, 32, 18, 2, 5, 2, 2)
    
    depthwise_conv2d_with_workload_nhwc(17, 32, 64, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(17, 32, 65, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(18, 32, 64, 2, 5, 1, 0)
    depthwise_conv2d_with_workload_nhwc(18, 32, 65, 2, 5, 1, 0)
    depthwise_conv2d_with_workload_nhwc(17, 32, 64, 1, 3, 2, 0)
    depthwise_conv2d_with_workload_nhwc(17, 32, 65, 1, 3, 2, 0)
    depthwise_conv2d_with_workload_nhwc(18, 32, 64, 2, 5, 2, 0)
    depthwise_conv2d_with_workload_nhwc(18, 32, 65, 2, 5, 2, 0)

if __name__ == "__main__":
    test_depthwise_conv2d()
