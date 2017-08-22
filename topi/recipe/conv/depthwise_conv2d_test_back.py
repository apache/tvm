import os
import tvm
import numpy as np
from scipy import signal
from tvm.contrib import nvcc

import topi
from topi.util import get_const_tuple
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_back_input_nhwc #, schedule_depthwise_conv2d_back_nhwc
from topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_back_weight_nhwc

import tensorflow as tf
TASK = "depthwise_conv2d"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", options=["-arch=sm_37"]) # 37 for k80(ec2 instance)
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

def test_depthwise_conv2d_back_input_nhwc():
    """You may test different settings."""
    batch = 3
    in_channel = 100
    in_height = 37
    in_width = 37

    channel_multiplier = 2
    filter_height = 3
    filter_width = 3

    stride_h = 1
    stride_w = 1

    padding_tf = 'VALID' # please change here accordingly
    padding_h = 0
    padding_w = 0

    out_height = np.int((in_height+2*padding_h-filter_height)/stride_h+1)
    out_width = np.int((in_width+2*padding_w-filter_width)/stride_w+1)
    out_channel = in_channel * channel_multiplier

    ishape = [batch, in_height, in_width, in_channel]
    oshape = [batch, out_height, out_width, out_channel]
    stride = [stride_h, stride_w]
    padding = [padding_h, padding_w]

    Out_grad = tvm.placeholder(oshape, name='Out_grad')
    Filter = tvm.placeholder((filter_height, filter_width, in_channel, channel_multiplier), name='Filter')

    In_grad = topi.nn.depthwise_conv2d_back_input_nhwc(Filter, Out_grad, oshape, ishape, stride, padding)

    schedule = schedule_depthwise_conv2d_back_input_nhwc(In_grad)

    f = tvm.build(schedule, [Filter, Out_grad, In_grad], 'cuda')
    ctx = tvm.gpu(0)

    # launch the kernel
    out_backprop_np = np.random.uniform(size=(batch, out_height, out_width, out_channel)).astype(Out_grad.dtype)
    filter_np = np.random.uniform(size=(filter_height, filter_width, in_channel, channel_multiplier)).astype(Filter.dtype)

    out_backprop_tvm = tvm.nd.array(out_backprop_np, ctx)
    filter_tvm = tvm.nd.array(filter_np, ctx)

    in_backprop_tvm = tvm.nd.array(np.zeros((batch, in_height, in_width, in_channel), dtype=Out_grad.dtype), ctx)

    f(filter_tvm, out_backprop_tvm, in_backprop_tvm)

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

    np.testing.assert_allclose(output_tf, in_backprop_tvm.asnumpy(), rtol=1e-5)
    print "success"

def test_depthwise_conv2d_back_filter_nhwc():
    batch = 3
    in_channel = 1024
    in_height = 37
    in_width = 37

    channel_multiplier = 2
    filter_height = 3
    filter_width = 3

    stride_h = 1
    stride_w = 1

    padding_h = 1
    padding_w = 1
    padding_tf = 'VALID' # please change here accordingly

    out_height = np.int((in_height+2*padding_h-filter_height)/stride_h+1)
    out_width = np.int((in_width+2*padding_w-filter_width)/stride_w+1)
    out_channel = in_channel * channel_multiplier

    fshape = [filter_height, filter_width, in_channel, channel_multiplier]
    oshape = [batch, out_height, out_width, out_channel]
    stride = [stride_h, stride_w]
    padding = [padding_h, padding_w]


    Out_grad = tvm.placeholder(oshape, name='Out_grad')
    Input = tvm.placeholder((batch, in_height, in_width, in_channel), name='In_grad')

    Weight_grad  = topi.nn.depthwise_conv2d_back_weight_nhwc(Input, Out_grad, oshape, fshape, stride, padding)

    schedule = schedule_depthwise_conv2d_back_weight_nhwc(Weight_grad)

    #print(tvm.lower(schedule,[Input, Out_grad, Weight_grad], simple_mode=True))
    f = tvm.build(schedule, [Input, Out_grad, Weight_grad], 'cuda')
    ctx = tvm.gpu(0)

    # launch the kernel
    out_backprop_np = np.random.uniform(size=(batch, out_height, out_width, out_channel)).astype(Out_grad.dtype)
    input_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(Input.dtype)

    out_backprop_tvm = tvm.nd.array(out_backprop_np, ctx)
    input_tvm = tvm.nd.array(input_np, ctx)

    weight_grad_tvm = tvm.nd.array(np.zeros((filter_height, filter_width, in_channel, channel_multiplier), dtype=Weight_grad.dtype), ctx)

    f(input_tvm, out_backprop_tvm, weight_grad_tvm)

    with tf.device('/gpu:0'):
        out_backprop_tf = tf.placeholder(tf.float32, [batch, out_height, out_width, out_channel])
        filter_shape_tf = tf.constant([filter_height, filter_width, in_channel, channel_multiplier])
        in_tf = tf.placeholder(tf.float32, [batch, in_height, in_width, in_channel])
        depth_conv_out = tf.nn.depthwise_conv2d_native_backprop_filter(input=in_tf,
                                                                      filter_sizes=filter_shape_tf,
                                                                      out_backprop=out_backprop_tf,
                                                                      strides=[1,stride_h,stride_w,1],
                                                                      padding='SAME')

        config = tf.ConfigProto()
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(tf.global_variables_initializer())
        output_tf = sess.run(depth_conv_out, feed_dict={out_backprop_tf:out_backprop_np, in_tf:input_np})

    np.testing.assert_allclose(output_tf, weight_grad_tvm.asnumpy(), rtol=1e-3)
    print "success"

if __name__ == "__main__":
    test_depthwise_conv2d_back_input_nhwc()
    test_depthwise_conv2d_back_filter_nhwc()
