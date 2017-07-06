"""Depthwise convolution operator.

Auto fusion with following elem-wise operators, e.g. batchnorm and relu.  
"""

import tvm
import os
import numpy as np
from tvm.contrib import nvcc_compiler
from topi.util import *

TASK="depthconv_map"
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

@tvm.tag_scope(tag="depthconv")
def depthconv(Input, Filter, Stride, padding):  
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
    stride_h = Stride.asnumpy()[0]
    stride_w = Stride.asnumpy()[1]

    # calculate output shape
    if padding == 'VALID':
        out_channel = in_channel * channel_multiplier
        out_height = (in_height - filter_height) / stride_h + 1
        out_width = (in_width - filter_width) / stride_w + 1
        pad_along_height = 0
        pad_along_width = 0
    if padding == 'SAME':
        out_channel = in_channel * channel_multiplier
        out_height = np.int(np.ceil(float(in_height) / float(stride_h)))
        out_width  = np.int(np.ceil(float(in_width) / float(stride_w)))
        pad_along_height = np.int(np.max((out_height - 1) * stride_h + filter_height - in_height, 0))
        pad_along_width = np.int(np.max((out_width - 1) * stride_w + filter_width - in_width, 0))
        
    height_after_pad = in_height + pad_along_height 
    width_after_pad = in_width + pad_along_width
    pad_top = np.int(np.ceil(float(pad_along_height) / 2))
    pad_left = np.int(np.ceil(float(pad_along_width) / 2))
    pad_bottom = pad_along_height - pad_top
    pad_right = pad_along_width - pad_left

    # padding stage
    PaddedInput = tvm.compute(
        (batch, in_channel, height_after_pad, width_after_pad),
        lambda b, c, i, j: tvm.select(
            tvm.all(i >= pad_top, i - pad_top < in_height, j >= pad_left, j - pad_left < in_width), 
            Input[b, c, i - pad_top, j - pad_left], tvm.const(0.0)), 
        name="PaddedInput")
    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    Output = tvm.compute(
        (batch, out_channel, out_height, out_width), 
        lambda b, c, i, j: tvm.sum(
            PaddedInput[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] * Filter[c/channel_multiplier, c%channel_multiplier, di, dj], 
            axis=[di, dj]),
        name='DepthConv')
    return Output

@tvm.tag_scope(tag="ewise")
def batchnorm(Input, BNparams):
    """Batch normalization operator.

    Parameters
    ----------
    Input : tvm.Tensor
    	Input tensor, layout is NCHW
    
    BNparams : tvm.Tensor
    	Value of scale and shift

    Returns
    -------
    Output : tvm.Tensor
    	Output tensor, layout is NCHW
    """
    Output = tvm.compute(Input.shape, lambda b, c, i, j: Input[b, c, i, j] * BNparams[c, 0] + BNparams[c, 1], name='BatchNorm')    
    return Output

@tvm.tag_scope(tag="ewise")
def relu(Input):
    """Relu operator.

    Parameters
    ----------
    Input : tvm.Tensor
    	Input tensor, layout is NCHW

    Returns
    -------
    Output : tvm.Tensor
    	Output tensor, layout is NCHW
    """
    Output = tvm.compute(Input.shape, lambda b, c, i, j: tvm.max(0, Input[b, c, i, j]), name='Relu')
    return Output

def schedule_depthconv_map(op):
    s = tvm.create_schedule(op)
    def schedule_depthconv(PaddedInput, Filter, DepthConv):
    	out_shape = get_const_tuple(DepthConv.shape)
        out_height = out_shape[2]
        out_width = out_shape[3]
        channel_multiplier = get_const_tuple(Filter.shape)[1]
        s[PaddedInput].compute_inline()
        IS = s.cache_read(PaddedInput, "shared", [DepthConv])
        FS = s.cache_read(Filter, "shared", [DepthConv]) 
        IL = s.cache_read(IS, "local", [DepthConv])
        FL = s.cache_read(FS, "local", [DepthConv])
        if is_output(DepthConv.op, s):
            Output = DepthConv
            CL = s.cache_write(DepthConv, "local")
        else:
            Output = op.output(0)
            s[DepthConv].set_scope("local")

        num_thread = 8
        num_vthread_x = 1
        num_vthread_y = 1
        blocking_h = 32
        blocking_w = 32
        if out_height % 48 == 0:
            blocking_h = 48
        if out_width % 48 == 0:
            blocking_w = 48
            num_vthread_y = 3
        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
        thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")

        bx, bxi = s[Output].split(Output.op.axis[1], factor=channel_multiplier)
        s[Output].reorder(Output.op.axis[2], Output.op.axis[3], bxi)
        bx = s[Output].fuse(bx, Output.op.axis[0])
        s[Output].bind(bx, block_x)
        by1, y1i = s[Output].split(Output.op.axis[2], factor=blocking_h)
        tvx, vxi = s[Output].split(y1i, nparts=num_vthread_x)
        tx, xi = s[Output].split(vxi, nparts=num_thread)
        by2, y2i = s[Output].split(Output.op.axis[3], factor=blocking_w)
        tvy, vyi = s[Output].split(y2i, nparts=num_vthread_y)
        ty, yi = s[Output].split(vyi, nparts=num_thread)
        s[Output].reorder(by1, by2, tvx, tvy, tx, ty, xi, yi)
        by = s[Output].fuse(by2, by1)
        s[Output].bind(tvx, thread_vx)
        s[Output].bind(tvy, thread_vy)
        s[Output].bind(tx, thread_x)
        s[Output].bind(ty, thread_y)
        s[Output].bind(by, block_y)

        s[IL].compute_at(s[Output], ty)
        s[FL].compute_at(s[Output], ty)
        if is_output(DepthConv.op, s):
            s[CL].compute_at(s[Output], ty)
        else:
            s[DepthConv].compute_at(s[Output], ty)
        # schedule for input's shared memory load
        s[IS].compute_at(s[Output], by)
        tx, xi = s[IS].split(IS.op.axis[2], nparts=num_thread)
        ty, yi = s[IS].split(IS.op.axis[3], nparts=num_thread)
        s[IS].bind(tx, thread_x)
        s[IS].bind(ty, thread_y)
        # schedule for filter's shared memory load
        s[FS].compute_at(s[Output], by)
        s[FS].reorder(FS.op.axis[2], FS.op.axis[3], FS.op.axis[1])  
        tx, xi = s[FS].split(FS.op.axis[2], nparts=num_thread)
        ty, yi = s[FS].split(FS.op.axis[3], nparts=num_thread)
        s[FS].bind(tx, thread_x)
        s[FS].bind(ty, thread_y)

    def traverse(OP):
        # inline all elem-wise operators except the last one (output)
        if is_ewise(OP):
            if not is_output(OP, s):  
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if not str(tensor.op.input_tensors) == str([]):
                    traverse(tensor.op)
        # schedule depthconv
        if is_depthconv(OP):
            PaddedInput = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            DepthConv = OP.output(0)
            schedule_depthconv(PaddedInput, Filter, DepthConv)

    traverse(op)
    return s
