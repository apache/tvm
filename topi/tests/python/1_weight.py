import tvm
import topi
import os
import numpy as np
import tensorflow as tf
from tvm.contrib import nvcc
from scipy import signal
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_back_weight_nhwc #, schedule_depthwise_conv2d_nchw

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

    oshape = [batch, out_height, out_width, out_channel]
    fshape = [filter_height, filter_width, in_channel, channel_multiplier]

    # tensorflow special
    padding_tf = 'SAME'
    if padding_h == 0:
        padding_tf = 'VALID'

    # placeholder
    Out_grad = tvm.placeholder(oshape, name='Out_grad')
    Input = tvm.placeholder((batch, in_height, in_width, in_channel), name='In_grad')
    stride = [stride_h, stride_w]
    padding = [padding_h, padding_w]

    # declare
    Weight_grad = topi.nn.depthwise_conv2d_back_weight_nhwc(Input, Out_grad, oshape, fshape, stride, padding)

    # schedule
    schedule = schedule_depthwise_conv2d_back_weight_nhwc(Weight_grad)
    #print(tvm.lower(schedule, [Input, Out_grad, Weight_grad], simple_mode=True))
    out_backprop_np = np.random.uniform(size=(batch, out_height, out_width, out_channel)).astype(Out_grad.dtype)
    input_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(Input.dtype)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        ctx = tvm.context(device, 0)

        # build the kernels
        f = tvm.build(schedule, [Input, Out_grad, Weight_grad], device)

        # prepare data
        out_backprop_tvm = tvm.nd.array(out_backprop_np, ctx)
        input_tvm = tvm.nd.array(input_np, ctx)

        weight_backprop_tvm = tvm.nd.array(np.zeros((filter_height, filter_width, in_channel, channel_multiplier), dtype=Weight_grad.dtype), ctx)

        # launch kernel (depthwise_conv2d backward nhwc wrt input)
        timer = f.time_evaluator(f.entry_name, ctx, number=0)
        tcost = timer(input_tvm, out_backprop_tvm, weight_backprop_tvm).mean
        ''' 
        # check with tensorflow
        with tf.device('/gpu:0'):
            out_backprop_tf = tf.placeholder(tf.float32, [batch, out_height, out_width, out_channel])
            filter_shape_tf = tf.constant([filter_height, filter_width, in_channel, channel_multiplier])
            in_tf = tf.placeholder(tf.float32, [batch, in_height, in_width, in_channel])
            depth_conv_out = tf.nn.depthwise_conv2d_native_backprop_filter(input=in_tf,
                                                                           filter_sizes=filter_shape_tf,
                                                                           out_backprop=out_backprop_tf,
                                                                           strides=[1,stride_h,stride_w,1],
                                                                           padding=padding_tf)

            config = tf.ConfigProto()
            sess = tf.Session(config=tf.ConfigProto())
            sess.run(tf.global_variables_initializer())
            output_tf = sess.run(depth_conv_out, feed_dict={out_backprop_tf:out_backprop_np, in_tf:input_np})
        '''
        Dilated_out_grad = topi.testing.dilate_python(out_backprop_np, [1, stride_h, stride_w, 1])
        if padding_h == 0 and padding_w == 0:
            output_np = np.zeros((filter_height, filter_width, in_channel, channel_multiplier))
            for c in range(in_channel):
                for m  in range(channel_multiplier):
                    for b in range(batch):
                        output_np[:, :, c, m] += signal.convolve2d(input_np[b, :, :, c], \
                            np.rot90(Dilated_out_grad[b, :, :, c*channel_multiplier+m%channel_multiplier], 2), \
                            mode='valid') #[padding_h:(padding_h+stride_h*filter_height):stride_h, padding_w:(padding_w+stride_w*filter_width):stride_w]

        if padding_h > 0 or padding_w > 0:
            output_np = np.zeros((filter_height, filter_width, in_channel, channel_multiplier))
            pad_top_tvm = np.int(np.ceil(float(padding_h) / 2))
            pad_left_tvm = np.int(np.ceil(float(padding_w) / 2))
            pad_top_scipy = np.int(np.ceil(float(filter_height - 1) / 2))
            pad_left_scipy = np.int(np.ceil(float(filter_width - 1) / 2))
            index_h = 7#(input_np.shape[1] - filter_height)/stride_h
            index_w = 7#(input_np.shape[2] - filter_width)/stride_w
            for c in range(in_channel):
                for m  in range(channel_multiplier):
                    for b in range(batch):
                         output_np[:,:,c,m] += signal.convolve2d(input_np[b, :, :, c], \
                            np.rot90(Dilated_out_grad[b, :, :, c*channel_multiplier+m%channel_multiplier], 2), \
                            mode='same')[index_h:(input_np.shape[1]-index_h), index_w:(input_np.shape[2]-index_w)]
        
        print("in_shape[%d,%d,%d,%d] filter[%d,%d,%d,%d] stride[%d,%d] padding[%d,%d] NHWC %.6f" %
                (batch, in_height, in_width, in_channel,
                 filter_height, filter_width, in_channel, channel_multiplier,
                 stride_h, stride_w, padding_h, padding_w,
                 tcost*1000))
        
        np.testing.assert_allclose(output_np, weight_backprop_tvm.asnumpy(), rtol=1e-4)
        print("success")

    check_device("cuda")

def test_depthwise_conv2d():
    print("testing nhwc")
    #depthwise_conv2d_with_workload_nhwc(17, 64, 17, 1, 3, 2, 1)
    #depthwise_conv2d_with_workload_nhwc(17, 32, 17, 1, 3, 2, 1)
    #depthwise_conv2d_with_workload_nhwc(18, 64, 17, 2, 5, 1, 2)
    #depthwise_conv2d_with_workload_nhwc(18, 32, 17, 2, 5, 1, 2)
    #depthwise_conv2d_with_workload_nhwc(17, 64, 17, 1, 3, 1, 0)
    #depthwise_conv2d_with_workload_nhwc(17, 32, 17, 1, 3, 1, 0)
    #depthwise_conv2d_with_workload_nhwc(18, 64, 17, 2, 5, 1, 0)
    #depthwise_conv2d_with_workload_nhwc(18, 32, 17, 2, 5, 1, 0)
    
    depthwise_conv2d_with_workload_nhwc(17, 64, 18, 1, 3, 2, 1)
    depthwise_conv2d_with_workload_nhwc(17, 32, 18, 1, 3, 2, 1)
    #depthwise_conv2d_with_workload_nhwc(18, 64, 18, 2, 5, 2, 2)
    #depthwise_conv2d_with_workload_nhwc(18, 32, 18, 2, 5, 2, 2)
    #depthwise_conv2d_with_workload_nhwc(17, 64, 18, 1, 3, 1, 0)
    #depthwise_conv2d_with_workload_nhwc(17, 32, 18, 1, 3, 1, 0)
    #depthwise_conv2d_with_workload_nhwc(18, 64, 18, 2, 5, 2, 0)
    #depthwise_conv2d_with_workload_nhwc(18, 32, 18, 2, 5, 2, 0)
if __name__ == "__main__":
    test_depthwise_conv2d()
