"""Depthwise convolution operator, with support for auto batchnorm + relu fusion."""

import tvm
import os
import numpy as np
from tvm.contrib import nvcc_compiler
from topi.util import get_const_tuple

TASK="depthwise_conv"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc_compiler.compile_source(code, target="ptx", options=["-arch=sm_52"])
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

def depthwise_conv(Input, Filter, Stride, padding, bn_relu_fusion=False, BNparams=None):  
    """Depthwise convolution operator.

    Parameters
    ----------
    Input : tvm.Tensor
    	4-D with shape [batch, in_channel, in_height, in_width] 

    Filter : tvm.Tensor
	4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]

    Stride : tvm.Tensor
	1-D of size 2
    
    padding : str 
	'VALID' or 'SAME'

    bn_relu_fusion : bool 
	Auto batchnorm + relu fusion when set True
    
    BNparams : tvm.Tensor
	2-D with shape [out_channel, 2], consists of scale and shift

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    in_shape = get_const_tuple(Input.shape)
    batch = in_shape[0]
    in_channel = in_shape[1]
    in_height = in_shape[2]
    in_width = in_shape[3]
    filter_shape = get_const_tuple(Filter.shape)
    filter_channel = filter_shape[0]
    channel_multiplier = filter_shape[1]
    filter_height = filter_shape[2]
    filter_width = filter_shape[3]
    if not in_height * in_width + channel_multiplier * filter_height * filter_width < 12000:
        print("Can't load input and filter into shared memory (exceeds 48K size limit)!")
        return
    stride_h = Stride.asnumpy()[0]
    stride_w = Stride.asnumpy()[1]

    if padding == 'VALID':
        # calculate output shape
        out_channel = in_channel * channel_multiplier
        out_height = (in_height - filter_height) / stride_h + 1
        out_width = (in_width - filter_width) / stride_w + 1
        # conv stage
        di = tvm.reduce_axis((0, filter_height), name='di')
        dj = tvm.reduce_axis((0, filter_width), name='dj')
        Conv = tvm.compute(
            (batch, out_channel, out_height, out_width), 
            lambda b, c, i, j: tvm.sum(
            	Input[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] * Filter[c/channel_multiplier, c%channel_multiplier, di, dj], 
            	axis=[di, dj]),
            name='Conv')

    if padding == 'SAME':
        # calculate output shape
        out_channel = in_channel * channel_multiplier
        out_height = np.int(np.ceil(float(in_height) / float(stride_h)))
        out_width  = np.int(np.ceil(float(in_width) / float(stride_w)))
        # padding stage
        pad_along_height = np.int(np.max((out_height - 1) * stride_h + filter_height - in_height, 0))
        pad_along_width = np.int(np.max((out_width - 1) * stride_w + filter_width - in_width, 0))
        height_after_pad = in_height + pad_along_height 
        width_after_pad = in_width + pad_along_width
        pad_top = np.int(np.ceil(float(pad_along_height) / 2))
        pad_left = np.int(np.ceil(float(pad_along_width) / 2))
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        PaddedInput = tvm.compute(
            (batch, in_channel, height_after_pad, width_after_pad),
            lambda b, c, i, j: tvm.select(
            	tvm.all(i >= pad_top, i - pad_top < in_height, j >= pad_left, j - pad_left < in_width), 
            	Input[b, c, i - pad_top, j - pad_left], tvm.const(0.0)), 
            name="PaddedInput")
        # conv stage
        di = tvm.reduce_axis((0, filter_height), name='di')
        dj = tvm.reduce_axis((0, filter_width), name='dj')
        Conv = tvm.compute(
            (batch, out_channel, out_height, out_width), 
            lambda b, c, i, j: tvm.sum(
            	PaddedInput[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] * Filter[c/channel_multiplier, c%channel_multiplier, di, dj], 
            	axis=[di, dj]),
            name='Conv')

    if bn_relu_fusion == True:
    	# batchnorm stage
        BatchNorm = tvm.compute(
            (batch, out_channel, out_height, out_width),
            lambda b, c, i, j: Conv[b, c, i, j] * BNparams[c, 0]+ BNparams[c, 1],
            name='BatchNorm') 
        # relu stage
        Relu = tvm.compute(
            (batch, out_channel, out_height, out_width),
            lambda b, c, i, j: tvm.max(BatchNorm[b, c, i, j], tvm.const(0.0)),
            name='Relu')
        Output = Relu
    else:
        Output = Conv

    # schedule
    s = tvm.create_schedule(Output.op)  
    if padding == 'VALID':
        IS = s.cache_read(Input, "shared", [Conv])
    if padding == 'SAME':
        IS = s.cache_read(PaddedInput, "shared", [Conv])
        s[PaddedInput].compute_inline()
    FS = s.cache_read(Filter, "shared", [Conv]) 
    IL = s.cache_read(IS, "local", [Conv])
    FL = s.cache_read(FS, "local", [Conv])
    
    if bn_relu_fusion == True:
        s[BatchNorm].compute_inline()
        s[Conv].set_scope("local")
        BL = s.cache_read(BNparams, "local", [BatchNorm])
    
    num_thread = 8
    num_vthread_x = 1
    num_vthread_y = 1
    if out_width % 16 == 0:
        num_vthread_y = out_width / 16     
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
    thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")

    s[Output].bind(Output.op.axis[0], block_y)
    bx, xi = s[Output].split(Output.op.axis[1], factor=channel_multiplier)
    s[Output].bind(bx, block_x) 
    s[Output].reorder(Output.op.axis[2], Output.op.axis[3], xi)  
    tvx, xi = s[Output].split(Output.op.axis[2], nparts=num_vthread_x)
    tx, xi = s[Output].split(xi, nparts=num_thread)
    tvy, yi = s[Output].split(Output.op.axis[3], nparts=num_vthread_y)
    ty, yi = s[Output].split(yi, nparts=num_thread)
    s[Output].bind(tvx, thread_vx)
    s[Output].bind(tvy, thread_vy)
    s[Output].bind(tx, thread_x)
    s[Output].bind(ty, thread_y)
    s[Output].reorder(tvx, tvy, tx, ty, xi, yi)

    if bn_relu_fusion == True:
        s[Conv].compute_at(s[Output], ty)
        s[BL].compute_at(s[Output], ty) 
 
    s[IL].compute_at(s[Output], ty)
    s[FL].compute_at(s[Output], ty)

    # schedule for input's shared memory load
    s[IS].compute_at(s[Output], bx)   
    tx, xi = s[IS].split(IS.op.axis[2], nparts=num_thread)
    ty, yi = s[IS].split(IS.op.axis[3], nparts=num_thread)
    s[IS].bind(tx, thread_x)
    s[IS].bind(ty, thread_y)
    # schedule for filter's shared memory load
    s[FS].compute_at(s[Output], bx)
    s[FS].reorder(FS.op.axis[2], FS.op.axis[3], FS.op.axis[1])  
    tx, xi = s[FS].split(FS.op.axis[2], nparts=num_thread)
    ty, yi = s[FS].split(FS.op.axis[3], nparts=num_thread)
    s[FS].bind(tx, thread_x)
    s[FS].bind(ty, thread_y)

    return s, Output
