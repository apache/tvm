import tvm
import topi
import numpy as np
from tvm.contrib.pickle_memoize import memoize
from scipy import signal
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_backward_input_nhwc

def depthwise_conv2d_with_workload_nhwc(batch, in_channel, in_height, channel_multiplier, filter_height, stride_h, padding_h):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_w = stride_h
    padding_w = padding_h

    out_height = np.int((in_height+2*padding_h-filter_height)/stride_h+1)
    out_width = np.int((in_width+2*padding_w-filter_width)/stride_w+1)
    out_channel = in_channel * channel_multiplier

    ishape = [batch, in_height, in_width, in_channel]
    oshape = [batch, out_height, out_width, out_channel]

    # placeholder
    Out_grad = tvm.placeholder(oshape, name='Out_grad')
    Filter = tvm.placeholder((filter_height, filter_width, filter_channel, channel_multiplier), name='Filter')
    stride = [stride_h, stride_w]
    padding = [padding_h, padding_w]

    # declare
    In_grad = topi.nn.depthwise_conv2d_backward_input_nhwc(Filter, Out_grad, oshape, ishape, stride, padding)

    # schedule
    schedule = schedule_depthwise_conv2d_backward_input_nhwc(In_grad)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.context(device, 0)

        # build the kernels
        f = tvm.build(schedule, [Filter, Out_grad, In_grad], device)

        # Use memoize, pickle the test data for next time use.
        @memoize("topi.tests.test_topi_depthwise_conv2d_backward_input")
        def get_ref_data():
            out_backprop_np = np.random.uniform(size=(batch, out_height, out_width, out_channel)).astype(Out_grad.dtype)
            filter_np = np.random.uniform(size=(filter_height, filter_width, in_channel, channel_multiplier)).astype(Filter.dtype)
            
            Dilated_out_grad = topi.testing.dilate_python(out_backprop_np, [1, stride_h, stride_w, 1]) 
            if padding_h > 0 and padding_w > 0:
                pad_top = int((filter_height - 1)/2)
                pad_left = int((filter_width - 1)/2)
                pad_bottom = int((filter_height - 1)/2)
                pad_right = int((filter_width - 1)/2)
            
                padded_out_grad = np.zeros((batch, Dilated_out_grad.shape[1]+pad_top+pad_bottom, Dilated_out_grad.shape[2]+pad_left+pad_right, out_channel))
                padded_out_grad[:, pad_top:-pad_bottom, pad_left:-pad_right, :] = Dilated_out_grad

                if stride_h % 2 == 0: # it pads an addition row to pad_right and pad_bottom
                    pad_top -= (in_height+stride_h+1)%2
                if stride_w % 2 == 0:
                    pad_left -= (in_width+stride_w+1)%2
                output_np = np.zeros((batch, in_height, in_width, in_channel))
                for b in range(batch):
                    for c in range(in_channel):
                        for m in range(channel_multiplier):
                            output_np[b, :, :, c] += signal.convolve2d(padded_out_grad[b, :, :, c*channel_multiplier+m], \
                            filter_np[:, :, c, m], \
                            mode='same')[(pad_top):-(pad_bottom), (pad_left):-(pad_right)] 
        
            if padding_h == 0 and padding_w == 0:
                pad_top = int((filter_height - 1)/2)
                pad_left = int((filter_width - 1)/2)
                pad_bottom = int((filter_height - 1)/2)
                pad_right = int((filter_width - 1)/2)
                if stride_h % 2 == 0: # it pads an addition row to pad_right and pad_bottom
                    pad_bottom += (in_height+stride_h+1)%2
                if stride_w % 2 == 0:
                    pad_right += (in_width+stride_w+1)%2
                padded_out_grad = np.zeros((batch, Dilated_out_grad.shape[1]+pad_top+pad_bottom, Dilated_out_grad.shape[2]+pad_left+pad_right, out_channel))
                padded_out_grad[:, pad_top:-pad_bottom, pad_left:-pad_right, :] = Dilated_out_grad
                output_np = np.zeros((batch, in_height, in_width, in_channel))
                for b in range(batch):
                    for c in range(in_channel):
                        for m in range(channel_multiplier):
                            output_np[b, :, :, c] += signal.convolve2d(padded_out_grad[b, :, :, c*channel_multiplier+m], \
                            filter_np[:, :, c, m], \
                            mode='same')
            
            return out_backprop_np, filter_np, output_np
        
        out_backprop_np, filter_np, output_np = get_ref_data()

        # prepare data
        out_backprop_tvm = tvm.nd.array(out_backprop_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)

        in_backprop_tvm = tvm.nd.array(np.zeros((batch, in_height, in_width, in_channel), dtype=Out_grad.dtype), ctx)

        # launch kernel (depthwise_conv2d backward nhwc wrt input)
        timer = f.time_evaluator(f.entry_name, ctx, number=1)
        tcost = timer(filter_tvm, out_backprop_tvm, in_backprop_tvm).mean

        
        print("in_shape[%d,%d,%d,%d] filter[%d,%d,%d,%d] stride[%d,%d] padding[%d,%d] NHWC %.6f" %
                (batch, in_height, in_width, in_channel,
                 filter_height, filter_width, in_channel, channel_multiplier,
                 stride_h, stride_w, padding_h, padding_w,
                 tcost*1000))
        np.testing.assert_allclose(output_np, in_backprop_tvm.asnumpy(), rtol=1e-5)
        print("success")
    check_device("cuda")

def test_depthwise_conv2d():
    print("testing nhwc")
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 1, 3, 1, 1)
    depthwise_conv2d_with_workload_nhwc(16, 256, 57, 1, 3, 1, 1)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 2, 5, 1, 2)
    depthwise_conv2d_with_workload_nhwc(16, 256, 57, 2, 5, 1, 2)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 1, 3, 2, 1)
    depthwise_conv2d_with_workload_nhwc(16, 256, 57, 1, 3, 2, 1)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 2, 5, 2, 2)
    depthwise_conv2d_with_workload_nhwc(16, 256, 57, 2, 5, 2, 2)
    
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 57, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 2, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 57, 2, 5, 1, 0) 
    depthwise_conv2d_with_workload_nhwc(16, 256, 55, 1, 3, 2, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 1, 3, 2, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 55, 2, 5, 2, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 2, 5, 2, 0)
    
if __name__ == "__main__":
    test_depthwise_conv2d()
