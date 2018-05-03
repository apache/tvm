# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""conv2d schedule on Intel GPU"""

from __future__ import absolute_import as _abs

import numpy as np
import tvm

from .. import generic
from .. import util
from .. import tag
from ..nn import pad
from ..nn.conv2d import conv2d
from ..nn.util import get_pad_tuple
from ..util import simplify


##### SCHEDULE UTILITIES #####
def fuse_and_bind(s, tensor, axis=None, num_thread=None):
    """ fuse all the axis and bind to GPU threads """
    axis = axis or s[tensor].op.axis
    fused = s[tensor].fuse(*axis)
    max_threads = tvm.target.current_target(allow_none=False).max_num_threads
    bx, tx = s[tensor].split(fused, num_thread or max_threads)
    s[tensor].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(tx, tvm.thread_axis("threadIdx.x"))
    return bx, tx

def split_and_bind(s, tensor, x, x_factor=1):
    bx, tx = s[tensor].split(x, factor = x_factor)
    s[tensor].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[tensor].bind(bx, tvm.thread_axis("blockIdx.x"))
    return bx, tx

def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """ tile and bind to GPU threads """
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi

def cache_tile_and_bind3d(s, tensor, z, y, x, z_factor = 2, y_factor=None, x_factor=None):
    """ tile and bind cache to GPU threads"""
    x_factor = x_factor or z_factor
    y_factor = y_factor or z_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].reorder(zo, yo, xo, zi, yi, xi)
    s[tensor].bind(zi, tvm.thread_axis("threadIdx.z"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    return zo, yo, xo, zi, yi, xi

def cache_tile_and_bind(s, tensor, y, x, y_factor=2, x_factor=None):
    """ tile and bind cache to GPU threads"""
    x_factor = x_factor or y_factor
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].reorder(yo, xo, yi, xi)
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    return yo, xo, yi, xi

def cache_split_and_bind(s, tensor, x, x_factor=1):
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    return xo

def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].reorder(zo, yo, xo ,zi, yi, xi)

    thread_z = tvm.thread_axis((0, z_factor), "threadIdx.z")
    thread_y = tvm.thread_axis((0, y_factor), "threadIdx.y")
    thread_x = tvm.thread_axis((0, x_factor), "threadIdx.x")
    s[tensor].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, thread_z)
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, thread_y)
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, thread_x)
    return xi, thread_z, thread_y, thread_x

@conv2d.register(["intel_gpu"])
def decl_conv2d(data, kernel, stride, padding, layout='NCHW', out_dtype='float32'):
    """Conv2D operator for Intel GPU backend.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert layout == 'NCHW', "only support NCHW convolution on intel gpu"
    assert data.shape[0].value == 1, "only support batch size=1 convolution on intel gpu"
    assert data.dtype == kernel.dtype, "Do not support inputs with different data types now."

    out_dtype = data.dtype
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    kernel_shape = util.get_const_tuple(kernel.shape)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    return _decl_cl_spatialpack(data, kernel, stride, padding, layout, out_dtype)

@generic.schedule_conv2d_nchw.register(["intel_gpu"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw for Intel GPU

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        if "4_5" in op.tag or "4_4" in op.tag or "2_7" in op.tag or "2_14" in op.tag or "1_16" in op.tag:
            _schedule_cl_spatialpack(s,op)

    traverse(outs[0].op)
    return s
    
def _decl_cl_spatialpack(data, kernel, stride, padding, layout, out_dtype='float16'):
    batch, in_channel, in_height, in_width = [util.get_const_int(x) for x in data.shape]
    num_filter, channel, kernel_h, kernel_w = [util.get_const_int(x) for x in kernel.shape]
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, kernel)

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride

    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    oshape = (batch, out_channel, out_height, out_width)
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(data, pad_before, pad_after, name="pad_temp")

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    block_w = 0
    block_h = 0
    if stride_h == 2:
        if num_filter + kernel_h == 515:
            conv_tag = "4_4"
            block_h = 4
            block_w = 4
        else:
            conv_tag = "4_5"
            block_h = 4
            block_w = 5
    elif kernel_h == 3:
        if num_filter == 512:
            conv_tag = "2_7"
            block_h = 2
            block_w = 7
        else:
            conv_tag = "2_14"
            block_h = 2
            block_w = 14
    else:
        conv_tag = "1_16"
        block_h = 1
        block_w = 16

    c_h = 0
    c_w = 0

    if out_height % block_h == 0:
        c_h = out_height
    else:
        c_h = (out_height // block_h + 1) * block_h

    if out_width % block_w == 0:
        c_w = out_width
    else:
        c_w = (out_width // block_w + 1) * block_w

    nv = 16
    cshape = (batch, out_channel // nv, c_h, c_w, nv)
    kvshape = (num_filter // nv, channel, kernel_h, kernel_w, nv)

    kernel_vec = tvm.compute(
            kvshape, 
            lambda co, ci, kh, kw, vc:
                kernel[co*nv + vc][ci][kh][kw], name='kernel_vec')

    conv = tvm.compute(
        cshape,
        lambda nn, ff, yy, xx, vc:\
          tvm.sum(
            temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx].astype(out_dtype) *
            kernel_vec[ff, rc, ry, rx, vc].astype(out_dtype),
            axis=[rc, ry, rx]), tag=conv_tag, name='conv')

    output = tvm.compute(
        oshape,
        lambda nn, ff, yy, xx:
            conv[nn][ff//nv][yy][xx][ff%nv],
            name='output_unpack', tag=conv_tag)

    return output

def _schedule_cl_spatialpack(s, op):
    output = op.output(0)
    _, _, out_height, out_width = [util.get_const_int(x) for x in output.shape]

    conv = op.input_tensors[0]
    temp = s[conv].op.input_tensors[0]
    kernel_vec = s[conv].op.input_tensors[1]
    kernel = s[kernel_vec].op.input_tensors[0]
    temp_W = s.cache_read(temp, "warp", [conv])
    conv_L = s.cache_write(conv, "local")

    kernel_L = s.cache_read(kernel_vec, "local", [conv_L])
    _, _, temp_h, temp_w = [util.get_const_int(x) for x in temp.shape]
    if "1_16" in s[conv].op.tag:
        OUTPUT_BLOCK_HEIGHT = 1
        OUTPUT_BLOCK_WIDTH  = 16
#        kernel_vec_tile = "fuse"
    elif "2_14" in s[conv].op.tag:
        OUTPUT_BLOCK_HEIGHT = 2
        OUTPUT_BLOCK_WIDTH  = 14
#        kernel_vec_tile = "fuse"
    elif "2_7" in s[conv].op.tag:
        OUTPUT_BLOCK_HEIGHT = 2
        OUTPUT_BLOCK_WIDTH  = 7
#        kernel_vec_tile = "inline"
    elif "4_5" in s[conv].op.tag:
        OUTPUT_BLOCK_HEIGHT = 4
        OUTPUT_BLOCK_WIDTH  = 5
#        kernel_vec_tile = "inline"
    elif "4_4" in s[conv].op.tag:
        OUTPUT_BLOCK_HEIGHT = 4
        OUTPUT_BLOCK_WIDTH  = 4
#        kernel_vec_tile = "fuse"

    # schedule conv
    z_factor = 1
    y_factor = 1
    x_factor = 16
    thread_z = tvm.thread_axis((0, z_factor), "threadIdx.z")
    thread_y = tvm.thread_axis((0, y_factor), "threadIdx.y")
    thread_x = tvm.thread_axis((0, x_factor), "threadIdx.x")
    _, co, oh, ow, vc = s[conv].op.axis
    ooh, ioh = s[conv].split(oh, factor = OUTPUT_BLOCK_HEIGHT)
    oow, iow = s[conv].split(ow, factor = OUTPUT_BLOCK_WIDTH)
    s[conv].reorder(_, co, ooh, oow, vc, ioh, iow)
    coo, coi = s[conv].split(co, nparts = 1)
    ooho, oohi = s[conv].split(ooh, factor = z_factor) 
    oowo, oowi = s[conv].split(oow, factor = y_factor)
    vco, vci = s[conv].split(vc, factor = x_factor)
    s[conv].reorder(_, coo, vco, ooho, oowo, coi, oohi, oowi, vci, ioh, iow)
    s[conv].bind(oohi, thread_z)
    s[conv].bind(oowi, thread_y)
    s[conv].bind(vci, thread_x)
    s[conv].bind(ooho, tvm.thread_axis("blockIdx.z"))
    s[conv].bind(oowo, tvm.thread_axis("blockIdx.y"))
    s[conv].bind(coi, tvm.thread_axis("blockIdx.x"))

    # schedule conv_L
    s[conv_L].compute_at(s[conv], vci)
    i, oc, h, w, vc = s[conv_L].op.axis
    rc, ry, rx = s[conv_L].op.reduce_axis
    s[conv_L].reorder(i, oc, rc, ry, rx, vc, h, w)
    if kernel.shape[3].value != 7:
        s[conv_L].unroll(ry)
        s[conv_L].unroll(rx)

    # schedule temp
    _, ci, h, w = s[temp].op.axis
    tile_and_bind3d(s, temp, ci,  h, w, 1, 16, 16)

    # schedule temp_W
    s[temp_W].compute_at(s[conv_L], rc)
    _, ci, h, w = s[temp_W].op.axis
    zo, zi = s[temp_W].split(ci, 1)
    yo, yi = s[temp_W].split(h, 1)
    xo, xi = s[temp_W].split(w, 16)
    s[temp_W].reorder(zo, yo, xo, zi, yi, xi)
    s[temp_W].bind(zi, thread_z)
    s[temp_W].bind(yi, thread_y)
    s[temp_W].bind(xi, thread_x)
    s[temp_W].storage_align(s[temp_W].op.axis[2], 16, 0)

    #schedule kernel
#    if kernel_vec_tile == "fuse":
#        fuse_and_bind(s, kernel_vec, num_thread = 256)
#    else:
    s[kernel_vec].compute_inline()

    # schedule kernel_L
    if "2_14" in s[conv].op.tag:
        s[kernel_L].compute_at(s[conv_L], ry)
    else:
        s[kernel_L].compute_at(s[conv_L], rx)

    # schedule output
    if output.op in s.outputs:
        out = output
    else:
        s[output].compute_inline()
        out = s.outputs[0]

    _, co, h, w = s[out].op.axis
    tile_and_bind3d(s, out, w, h, co, 4, 8, 8)

