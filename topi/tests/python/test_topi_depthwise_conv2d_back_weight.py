import tvm
import topi
import numpy as np
import tensorflow as tf
from scipy import signal
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_back_weight_nhwc #, schedule_depthwise_conv2d_nchw

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
    # schedule = schedule_depthwise_conv2d_back_weight_nhwc(Weight_grad)
    schedule = tvm.create_schedule(Weight_grad.op)

    out_backprop_np = np.random.uniform(size=(batch, out_height, out_width, out_channel)).astype(Out_grad.dtype)
    input_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(Input.dtype)

    def check_device(device):
        # if not tvm.module.enabled(device):
        #     print("Skip because %s is not enabled" % device)
        #     return
        ctx = tvm.cpu(0)

        # build the kernels
        f = tvm.build(schedule, [Input, Out_grad, Weight_grad], 'llvm')

        # prepare data
        out_backprop_tvm = tvm.nd.array(out_backprop_np, ctx)
        input_tvm = tvm.nd.array(input_np, ctx)

        weight_backprop_tvm = tvm.nd.array(np.zeros((filter_height, filter_width, in_channel, channel_multiplier), dtype=Weight_grad.dtype), ctx)

        # launch kernel (depthwise_conv2d backward nhwc wrt input)
        timer = f.time_evaluator(f.entry_name, ctx, number=1)
        tcost = timer(input_tvm, out_backprop_tvm, weight_backprop_tvm).mean

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

        print("in_shape[%d,%d,%d,%d] filter[%d,%d,%d,%d] stride[%d,%d] padding[%d,%d] NHWC %.6f" %
                (batch, in_height, in_width, in_channel,
                 filter_height, filter_width, in_channel, channel_multiplier,
                 stride_h, stride_w, padding_h, padding_w,
                 tcost*1000))

        np.testing.assert_allclose(output_tf, weight_backprop_tvm.asnumpy(), rtol=1e-5)
        print("success")

    check_device("cuda")

def test_depthwise_conv2d():
    print("testing nhwc")
    depthwise_conv2d_with_workload_nhwc(1, 728, 64, 1, 3, 1, 1)
    depthwise_conv2d_with_workload_nhwc(1, 728, 32, 1, 3, 1, 1)
    depthwise_conv2d_with_workload_nhwc(4, 256, 65, 2, 5, 2, 2) # stride should be 2
    depthwise_conv2d_with_workload_nhwc(4, 256, 33, 2, 5, 2, 2) # stride should be 2
    depthwise_conv2d_with_workload_nhwc(1, 728, 64, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(1, 728, 32, 1, 3, 1, 0)
    depthwise_conv2d_with_workload_nhwc(4, 256, 65, 2, 5, 2, 0)
    depthwise_conv2d_with_workload_nhwc(4, 256, 33, 2, 5, 2, 0)

if __name__ == "__main__":
    test_depthwise_conv2d()
