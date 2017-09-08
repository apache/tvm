import tvm
import topi
import numpy as np
import tensorflow as tf
import os
from tvm.contrib import nvcc
from scipy import signal
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_back_input_nhwc #, schedule_depthwise_conv2d_nchw

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

@tvm.register_func
def tvm_callback_cuda_compile(code):
        ptx = nvcc.compile_cuda(code, target="ptx", options=["-arch=sm_37"]) # 37 for k80(ec2 instance)
        return ptx

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

    # tensorflow special
    padding_tf = 'SAME'
    if padding_h == 0:
        padding_tf = 'VALID'

    # placeholder
    Out_grad = tvm.placeholder(oshape, name='Out_grad')
    Filter = tvm.placeholder((filter_height, filter_width, filter_channel, channel_multiplier), name='Filter')
    stride = [stride_h, stride_w]
    padding = [padding_h, padding_w]

    # declare
    In_grad = topi.nn.depthwise_conv2d_back_input_nhwc(Filter, Out_grad, oshape, ishape, stride, padding)

    # schedule
    schedule = schedule_depthwise_conv2d_back_input_nhwc(In_grad)

    out_backprop_np = np.random.uniform(size=(batch, out_height, out_width, out_channel)).astype(Out_grad.dtype)
    filter_np = np.random.uniform(size=(filter_height, filter_width, in_channel, channel_multiplier)).astype(Filter.dtype)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.context(device, 0)

        # build the kernels
        f = tvm.build(schedule, [Filter, Out_grad, In_grad], device)

        # prepare data
        out_backprop_tvm = tvm.nd.array(out_backprop_np, ctx)
        filter_tvm = tvm.nd.array(filter_np, ctx)

        in_backprop_tvm = tvm.nd.array(np.zeros((batch, in_height, in_width, in_channel), dtype=Out_grad.dtype), ctx)

        # launch kernel (depthwise_conv2d backward nhwc wrt input)
        timer = f.time_evaluator(f.entry_name, ctx, number=1)
        tcost = timer(filter_tvm, out_backprop_tvm, in_backprop_tvm).mean
        '''
        # correctness with tensorflow
        with tf.device('/gpu:0'):
            out_backprop_tf = tf.placeholder(tf.float32, oshape)
            filter_tf = tf.placeholder(tf.float32, [filter_height, filter_width, in_channel, channel_multiplier])
            In_shape_tf = tf.constant([batch, in_height, in_width, in_channel])
            depth_conv_out = tf.nn.depthwise_conv2d_native_backprop_input(input_sizes=In_shape_tf,
                                                                          filter=filter_tf,
                                                                          out_backprop=out_backprop_tf,
                                                                          strides=[1,stride_h,stride_w,1],
                                                                          padding=padding_tf)
  
            config = tf.ConfigProto()
            sess = tf.Session(config=tf.ConfigProto())
            sess.run(tf.global_variables_initializer())
            output_tf = sess.run(depth_conv_out, feed_dict={out_backprop_tf:out_backprop_np, filter_tf:filter_np})
        '''
        Dilated_out_grad = topi.testing.dilate_python(out_backprop_np, [1, stride_h, stride_w, 1])
        
        if padding_h == 0 and padding_w == 0:
            pad_top = (filter_height - 1)/2
            pad_left = (filter_width - 1)/2
            pad_bottom = (filter_height - 1)/2 + (in_height+stride_h)%2
            pad_right = (filter_width - 1)/2  + (in_width+stride_w)%2
            padded_out_grad = np.zeros((batch, Dilated_out_grad.shape[1]+pad_top+pad_bottom, Dilated_out_grad.shape[2]+pad_left+pad_right, out_channel))
            padded_out_grad[:, pad_top:-pad_bottom, pad_left:-pad_right, :] = Dilated_out_grad
            output_np = np.zeros((batch, in_height, in_width, in_channel))
            for b in range(batch):
                for c in range(in_channel):
                    for m in range(channel_multiplier):
                        output_np[b, :, :, c] += signal.convolve2d(padded_out_grad[b, :, :, c*channel_multiplier+m], \
                        filter_np[:, :, c, m], \
                        mode='same')#[pad_top:, pad_left:] 
        
        print("in_shape[%d,%d,%d,%d] filter[%d,%d,%d,%d] stride[%d,%d] padding[%d,%d] NHWC %.6f" %
                (batch, in_height, in_width, in_channel,
                 filter_height, filter_width, in_channel, channel_multiplier,
                 stride_h, stride_w, padding_h, padding_w,
                 tcost*1000))
        #print("@@@@", output_np, "!!!!",in_backprop_tvm.asnumpy())
        np.testing.assert_allclose(output_np, in_backprop_tvm.asnumpy(), rtol=1e-5)
        #np.testing.assert_allclose(output_tf, in_backprop_tvm.asnumpy(), rtol=1e-5)
        print("success")
    check_device("cuda")

def test_depthwise_conv2d():
    print("testing nhwc")
    #depthwise_conv2d_with_workload_nhwc(64, 728, 64, 1, 3, 1, 1)
    #depthwise_conv2d_with_workload_nhwc(64, 728, 32, 1, 3, 1, 1)
    #depthwise_conv2d_with_workload_nhwc(64, 256, 64, 2, 5, 1, 2)
    #depthwise_conv2d_with_workload_nhwc(64, 256, 32, 2, 5, 1, 2)
    depthwise_conv2d_with_workload_nhwc(16, 256, 72, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 73, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 72, 2, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 73, 2, 5, 1, 0)
    
    depthwise_conv2d_with_workload_nhwc(16, 256, 55, 1, 3, 2, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 1, 3, 2, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 55, 2, 5, 2, 0)
    depthwise_conv2d_with_workload_nhwc(16, 256, 56, 2, 5, 2, 0)

if __name__ == "__main__":
    test_depthwise_conv2d()
