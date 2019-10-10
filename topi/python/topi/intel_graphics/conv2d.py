# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return, too-many-arguments, too-many-locals, too-many-statements, no-member, too-many-branches, too-many-boolean-expressions
"""conv2d schedule on Intel Graphics"""

from __future__ import absolute_import as _abs

import tvm

from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from ..nn.conv2d import conv2d, conv2d_NCHWc, conv2d_alter_layout, conv2d_infer_layout
from ..nn.util import get_pad_tuple
from ..nn.depthwise_conv2d import depthwise_conv2d_nchw
from ..nn import pad
from .. import tag
from .. import generic
from .. import util
from ..util import simplify, get_const_tuple


def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False):
    if is_depthwise:
        raise RuntimeError("Depthwise not supported for intel graphics.")
    else:
        batch_size, in_channel, height, width = get_const_tuple(data.shape)
        out_channel, _, hkernel, _ = get_const_tuple(kernel.shape)
        HSTR, _ = strides

        ic_bn = 1
        oc_bn, oc_bn_upper = 16, 16
        for i in range(oc_bn_upper, 0, -1):
            if out_channel % i == 0:
                oc_bn = i
                break

        if HSTR == 2:
            if out_channel + hkernel == 515:
                block_oh = 4
                block_ow = 4
            else:
                block_oh = 4
                block_ow = 5
        elif hkernel == 3:
            if out_channel == 512:
                block_oh = 2
                block_ow = 7
            else:
                block_oh = 2
                block_ow = 14
        else:
            block_oh = 1
            block_ow = 16
        cfg["tile_ic"] = SplitEntity([in_channel // ic_bn, ic_bn])
        cfg["tile_oc"] = SplitEntity([out_channel // oc_bn, oc_bn])
        cfg["block_oh"] = OtherOptionEntity(block_oh)
        cfg["block_ow"] = OtherOptionEntity(block_ow)


def _create_schedule_template(cfg, data, kernel, strides, padding, dilation, layout):
    """Create schedule configuration from input arguments"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    if layout == 'NCHW':
        n, ic, h, w = dshape
        oc, _, kh, kw = kshape
    else:
        raise ValueError("Not support this layout {} with "
                         "schedule template.".format(layout))
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (h - kh + 2 * ph) // sh + 1
    ow = (w - kw + 2 * pw) // sw + 1
    ic_bn_upper = 32
    oc_bn_upper = 64
    oc_bn_lower = min(oc, 8)
    ic_bn_candidates, oc_bn_candidates = [], []
    for i in range(1, ic + 1):
        if ic % i == 0 and i <= ic_bn_upper:
            ic_bn_candidates.append(i)
    if not ic_bn_candidates:
        ic_bn_candidates.append(1)
        ic_bn_candidates.append(ic)

    for i in range(1, oc + 1):
        if oc % i == 0 and oc_bn_lower <= i <= oc_bn_upper:
            oc_bn_candidates.append(i)
    if not oc_bn_candidates:
        oc_bn_candidates.append(1)
        oc_bn_candidates.append(oc)

    blk_candidates_low_limits = 5
    blk_oh_list, blk_ow_list = [], []
    for i, j in zip(range(oh, 0, -1), range(ow, 0, -1)):
        if i <= 16 and oh % i == 0:
            blk_oh_list.append(i)
        if j <= 16 and ow % j == 0:
            blk_ow_list.append(j)

    if len(blk_oh_list) < blk_candidates_low_limits:
        for i in range(2, oh):
            if i not in blk_oh_list:
                blk_oh_list.append(i)
                if len(blk_oh_list) >= 5:
                    break

    if len(blk_ow_list) < blk_candidates_low_limits:
        for i in range(min(ow - 1, 16), 1, -1):
            if i not in blk_ow_list:
                blk_ow_list.append(i)
                if len(blk_ow_list) >= 5:
                    break

    # Create schedule config
    cfg.define_knob("tile_ic", ic_bn_candidates)
    cfg.define_knob("tile_oc", oc_bn_candidates)
    cfg.define_knob("block_oh", blk_oh_list)
    cfg.define_knob("block_ow", blk_ow_list)


##### SCHEDULE UTILITIES #####
def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].reorder(zo, yo, xo, zi, yi, xi)

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

# Define template function for autotvm task
# We define schedule template in this function instead of
# declaration function since actual input arguments need
# to be altered by the schedule selected.
@autotvm.task.register("topi_intel_graphics_conv2d_NCHWc")
def __topi_nn_conv2d_NCHWc(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    data, kernel, strides, padding, dilation, layout, dtype = deserialize_args(args)
    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_schedule_template(cfg, data, kernel, strides, padding, dilation, layout)
    cfg.add_flop(1)

    # change shape with the value in config
    ic_bn = cfg["tile_ic"].val if hasattr(cfg["tile_ic"], "val") else cfg["tile_ic"].size[-1]
    oc_bn = cfg["tile_oc"].val if hasattr(cfg["tile_oc"], "val") else cfg["tile_oc"].size[-1]

    new_data_shape = (raw_data_shape[0], raw_data_shape[1] // ic_bn,
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    new_kernel_shape = (raw_kernel_shape[0] // oc_bn, raw_kernel_shape[1] // ic_bn,
                        raw_kernel_shape[2], raw_kernel_shape[3], ic_bn, oc_bn)
    new_data = tvm.placeholder(new_data_shape, data.dtype)
    new_kernel = tvm.placeholder(new_kernel_shape, kernel.dtype)

    C = _decl_cl_spatialpack_NCHWc(cfg, new_data, new_kernel, strides, padding, dilation, dtype)
    s = _schedule_conv2d_NCHWc(cfg, [C])

    return s, [new_data, new_kernel, C]

@conv2d_alter_layout.register(["intel_graphics"])
def _alter_conv2d_layout(attrs, inputs, tinfo, F):
    import nnvm.symbol as sym

    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}

    if F.__name__ == 'tvm.relay.op':
        # Derive channels for frontends (e.g ONNX) that miss "channel" field.
        new_attrs["channels"] = inputs[1].checked_type.shape[attrs['kernel_layout'].index('O')]

    data, kernel = tinfo[0], tinfo[1]
    batch_size, in_channel, height, width = get_const_tuple(data.shape)

    groups = attrs.get_int("groups")
    out_channel = attrs.get_int("channels")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    out_dtype = attrs["out_dtype"]

    layout_name = 'layout' if F == sym else 'data_layout'
    layout = attrs[layout_name]
    kh, kw = attrs.get_int_tuple("kernel_size")

    dtype = data.dtype
    out_dtype = dtype if out_dtype in ("same", "") else out_dtype
    is_depthwise = groups == in_channel and groups == out_channel

    # only optimize for NCHW
    if layout != 'NCHW':
        return None
    if groups != 1 and not is_depthwise:
        return None

    dispatch_ctx = autotvm.task.DispatchContext.current
    target = tvm.target.current_target()

    # query schedule and fallback if necessary
    workload = autotvm.task.args_to_workload(
        [data, kernel, strides, padding, dilation, out_dtype], depthwise_conv2d_nchw) \
        if is_depthwise else \
        autotvm.task.args_to_workload(
            [data, kernel, strides, padding, dilation, layout, out_dtype], conv2d)
    if is_depthwise:
        return None
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:
        _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise)

    ic_bn = cfg["tile_ic"].val if hasattr(cfg["tile_ic"], "val") else cfg["tile_ic"].size[-1]
    oc_bn = cfg["tile_oc"].val if hasattr(cfg["tile_oc"], "val") else cfg["tile_oc"].size[-1]

    new_attrs[layout_name] = 'NCHW%dc' % ic_bn
    new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

    new_data = tvm.placeholder((batch_size, in_channel//ic_bn, height, width, ic_bn),
                               dtype=data.dtype)

    out_channel, _, kh, kw = get_const_tuple(kernel.shape)
    # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
    new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)

    # Store altered operator's config
    new_kernel = tvm.placeholder((out_channel//oc_bn, in_channel//ic_bn, kh, kw, ic_bn, oc_bn),
                                 dtype=kernel.dtype)
    new_workload = autotvm.task.args_to_workload(
        [new_data, new_kernel, strides, padding, dilation, new_attrs[layout_name],
         new_attrs['out_layout'], out_dtype], conv2d_NCHWc)

    dispatch_ctx.update(target, new_workload, cfg)
    if F == sym:
        return F.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
    return F.nn.contrib_conv2d_nchwc(*copy_inputs, **new_attrs)

@autotvm.register_topi_compute(conv2d_NCHWc, 'intel_graphics', 'direct')
def _decl_conv2d(cfg, data, kernel, strides, padding, dilation,
                 layout, out_layout, out_dtype='float32'):
    """Conv2D operator for Intel Graphics backend.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        5-D with shape [num_filter, in_channel, filter_height, filter_width, nnum_filter_vec]

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
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    assert (dh, dw) == (1, 1), "Does not support dilation"

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    oc_chunk, _, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
    in_channel = ic_chunk * ic_bn
    num_filter = oc_chunk * oc_bn
    if cfg.is_fallback:
        _get_default_config(cfg, tvm.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
                            tvm.placeholder((num_filter, in_channel, kernel_height, kernel_width),
                                            dtype=kernel.dtype),
                            strides, padding, out_dtype)

    return _decl_cl_spatialpack_NCHWc(cfg, data, kernel, strides, padding, dilation, out_dtype)


@conv2d_infer_layout.register("intel_graphics")
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, dtype = workload
    batch_size, in_channel, in_height, in_width = data[:-1]
    out_channel, _, k_height, k_width = kernel[:-1]
    out_height = (in_height + 2 * padding[0] - k_height) // strides[0] + 1
    out_width = (in_width + 2 * padding[1] - k_width) // strides[1] + 1
    tile_ic, tile_oc = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    in_shape = (batch_size, in_channel // tile_ic, in_height, in_width, tile_ic)
    in_layout = "NCHW%dc" % tile_ic
    out_shape = (batch_size, out_channel // tile_oc, out_height, out_width, tile_oc)
    out_layout = "NCHW%dc" % tile_oc
    return ((in_shape, in_layout),), ((out_shape, out_layout),)


@autotvm.register_topi_schedule(generic.schedule_conv2d_NCHWc, 'intel_graphics', ['direct'])
def _schedule_conv2d_NCHWc(cfg, outs):
    """Schedule for conv2d_nchw for Intel Graphics

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
    scheduled_ops = []

    def traverse(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        if "conv" in op.tag:
            _schedule_cl_spatialpack_NCHWc(cfg, s, op)

        scheduled_ops.append(op)

    traverse(outs[0].op)

    return s

def _decl_cl_spatialpack_NCHWc(cfg, data, kernel, strides, padding, dilation, out_dtype='float16'):
    batch, in_channel, in_height, in_width, vc = [util.get_const_int(x) for x in data.shape]
    in_channel *= vc
    num_filter, channel, kernel_h, kernel_w, ci, co = [util.get_const_int(x) for x in kernel.shape]
    num_filter *= co
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, kernel)

    ic_bn = vc
    assert vc == ci

    if isinstance(strides, (tuple, list)):
        stride_h, stride_w = strides
    else:
        stride_h, stride_w = strides, strides

    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    oshape = (batch, out_channel // co, out_height, out_width, co)

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    block_h = cfg["block_oh"].val
    block_w = cfg["block_ow"].val

    c_h = out_height
    c_w = out_width

    if out_height % block_h != 0:
        c_h = (out_height // block_h + 1) * block_h

    if out_width % block_w != 0:
        c_w = (out_width // block_w + 1) * block_w

    cshape = (batch, out_channel // co, c_h, c_w, co)

    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down + c_h - out_height, pad_right + \
                 c_w - out_width, 0]
    DOPAD = (pad_top != 0 or pad_left != 0 or pad_down + c_h - out_height != 0 \
             or pad_right + c_w - out_width != 0)
    DOUNPACK = (c_h - out_height != 0 or c_w - out_width != 0)
    if DOPAD:
        temp = pad(data, pad_before, pad_after, name="pad_temp")
    else:
        temp = data

    conv = tvm.compute(
        cshape,
        lambda nn, ff, yy, xx, ff_v: \
            tvm.sum(
                temp[nn, rc//ic_bn, yy * stride_h + ry, xx * stride_w + rx, rc%ic_bn]. \
                        astype(out_dtype) *
                kernel[ff, rc//ic_bn, ry, rx, rc%ic_bn, ff_v].astype(out_dtype),
                axis=[rc, ry, rx]), tag="conv", name='conv')

    if DOUNPACK:
        output = tvm.compute(
            oshape,
            lambda nn, ff, yy, xx, ff_v:
            conv[nn][ff][yy][xx][ff_v],
            name='output_unpack', tag="conv_unpack")
    else:
        output = conv


    return output


def _schedule_cl_spatialpack_NCHWc(cfg, s, op):
    output = op.output(0)
    conv = op.input_tensors[0]
    if conv.op.name == "conv":
        temp = s[conv].op.input_tensors[0]
        kernel = s[conv].op.input_tensors[1]
        temp_W = s.cache_read(temp, "warp", [conv])
        conv_L = s.cache_write(conv, "local")
        SCHEDULE_OUTPUT = True
    else:
        temp = op.input_tensors[0]
        kernel = op.input_tensors[1]
        temp_W = s.cache_read(temp, "warp", [output])
        conv_L = s.cache_write(output, "local")
        if output.op in s.outputs:
            conv = output
        else:
            s[output].compute_inline()
            conv = s.outputs[0]
        SCHEDULE_OUTPUT = False
    kernel_L = s.cache_read(kernel, "local", [conv_L])

    OUTPUT_BLOCK_HEIGHT = cfg["block_oh"].val
    OUTPUT_BLOCK_WIDTH = cfg["block_ow"].val

    # schedule conv
    z_factor = 1
    y_factor = 1
    x_factor = 16
    thread_z = tvm.thread_axis((0, z_factor), "threadIdx.z")
    thread_y = tvm.thread_axis((0, y_factor), "threadIdx.y")
    thread_x = tvm.thread_axis((0, x_factor), "threadIdx.x")
    _, co, oh, ow, vc = s[conv].op.axis
    ooh, ioh = s[conv].split(oh, factor=OUTPUT_BLOCK_HEIGHT)
    oow, iow = s[conv].split(ow, factor=OUTPUT_BLOCK_WIDTH)
    s[conv].reorder(_, co, ooh, oow, vc, ioh, iow)
    coo, coi = s[conv].split(co, nparts=1)
    ooho, oohi = s[conv].split(ooh, factor=z_factor)
    oowo, oowi = s[conv].split(oow, factor=y_factor)
    vco, vci = s[conv].split(vc, factor=x_factor)
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
    s[temp_W].compute_at(s[conv_L], rc)
    if kernel.shape[3].value != 7:
        s[conv_L].unroll(ry)
        s[conv_L].unroll(rx)

    # schedule temp
    if temp.op.name == "pad_temp":
        _, ci, h, w, vci = s[temp].op.axis
        tile_and_bind3d(s, temp, ci, h, w, 1, 16, 16)

    # schedule temp_W
    _, ci, h, w, vci = s[temp_W].op.axis
    zo, zi = s[temp_W].split(vci, 1)
    yo, yi = s[temp_W].split(h, 1)
    xo, xi = s[temp_W].split(w, 16)
    s[temp_W].reorder(zo, yo, xo, zi, yi, xi)
    s[temp_W].bind(zi, thread_z)
    s[temp_W].bind(yi, thread_y)
    s[temp_W].bind(xi, thread_x)
    s[temp_W].storage_align(s[temp_W].op.axis[2], 16, 0)

    # schedule kernel_L
    if OUTPUT_BLOCK_HEIGHT == 2 and OUTPUT_BLOCK_WIDTH == 14:
        s[kernel_L].compute_at(s[conv_L], ry)
    else:
        s[kernel_L].compute_at(s[conv_L], rx)

    # schedule output
    if SCHEDULE_OUTPUT:
        if output.op in s.outputs:
            out = output
        else:
            s[output].compute_inline()
            out = s.outputs[0]

        _, co, h, w, vc = s[out].op.axis
        tile_and_bind3d(s, out, w, h, vc, 4, 8, 8)


def conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype):
    """convert argument to workload"""
    if len(kernel.shape) == 4:
        raw_kernel = kernel
    else:  # the input kernel is transformed by alter_op_layout
        shape = get_const_tuple(kernel.shape)
        raw_kernel = tvm.placeholder((shape[0] * shape[4], shape[1], shape[2], shape[3]),
                                     dtype=kernel.dtype)
    return ('conv2d', ) + autotvm.task.args_to_workload(
        [data, raw_kernel, strides, padding, layout, out_dtype])

@autotvm.register_topi_compute(conv2d, 'intel_graphics', 'direct')
def decl_conv2d(cfg, data, kernel, stride, padding, dilation, layout='NCHW', out_dtype='float32'):
    """Conv2D operator for Intel Graphics backend.

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

    return _decl_cl_spatialpack(cfg, data, kernel, stride, padding, layout, out_dtype)

@autotvm.task.register_topi_schedule(generic.schedule_conv2d_nchw, 'intel_graphics', ['direct'])
def schedule_conv2d_nchw(cfg, outs):
    """Schedule for conv2d_nchw for Intel Graphics

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
    scheduled_ops = []

    def traverse(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        if 'conv2d' in op.tag:
            _schedule_cl_spatialpack(cfg, s, op)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

def _decl_cl_spatialpack(cfg, data, kernel, stride, padding, layout, out_dtype='float16'):
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

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    block_w = 1
    block_h = 1
    if stride_h == 2:
        if num_filter + kernel_h == 515:
            block_h = 4
            block_w = 4
        else:
            block_h = 4
            block_w = 5
    elif kernel_h == 3:
        if num_filter == 512:
            block_h = 2
            block_w = 7
        else:
            block_h = 2
            block_w = 14
    elif kernel_h == 7 and padding == 3 and stride == 1:
        block_h = 3
        block_w = 4
    else:
        block_h = 1
        block_w = 16
    attrs = {'block_h': block_h, 'block_w' : block_w}
    c_h = out_height
    c_w = out_width

    if out_height % block_h != 0:
        c_h = (out_height // block_h + 1) * block_h

    if out_width % block_w != 0:
        c_w = (out_width // block_w + 1) * block_w

    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down + c_h - block_h, pad_right + c_w - block_w]
    temp = pad(data, pad_before, pad_after, name="pad_temp")

    nv = 16
    if num_filter % nv != 0:
        num_filter = (num_filter // nv + 1) * nv
        out_channel = num_filter

    cshape = (batch, out_channel // nv, c_h, c_w, nv)
    kvshape = (num_filter // nv, channel, kernel_h, kernel_w, nv)

    kernel_vec = tvm.compute(
        kvshape,
        lambda co, ci, kh, kw, vc:
        kernel[co*nv + vc][ci][kh][kw], name='kernel_vec')

    conv = tvm.compute(
        cshape,
        lambda nn, ff, yy, xx, vc: \
            tvm.sum(
                temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx].astype(out_dtype) *
                kernel_vec[ff, rc, ry, rx, vc].astype(out_dtype),
                axis=[rc, ry, rx]), name='conv', attrs=attrs)

    output = tvm.compute(
        oshape,
        lambda nn, ff, yy, xx:
        conv[nn][ff//nv][yy][xx][ff%nv],
        name='output_unpack', tag='conv2d',
        attrs={'workload': conv_arg_to_workload(data, kernel, stride, padding,
                                                layout, out_dtype)})

    return output

def _schedule_cl_spatialpack(cfg, s, op):
    output = op.output(0)
    _, _, out_height, out_width = [util.get_const_int(x) for x in output.shape]

    conv = op.input_tensors[0]
    temp = s[conv].op.input_tensors[0]
    kernel_vec = s[conv].op.input_tensors[1]
    kernel = s[kernel_vec].op.input_tensors[0]
    temp_W = s.cache_read(temp, "warp", [conv])
    conv_L = s.cache_write(conv, "local")

    kernel_L = s.cache_read(kernel_vec, "local", [conv_L])
    _, in_channel, temp_h, temp_w = [util.get_const_int(x) for x in temp.shape]

    attrs = s[conv].op.attrs
    OUTPUT_BLOCK_HEIGHT = attrs['block_h']
    OUTPUT_BLOCK_WIDTH = attrs['block_w']

    # schedule conv
    z_factor = 1
    y_factor = 1
    x_factor = 16
    thread_z = tvm.thread_axis((0, z_factor), "threadIdx.z")
    thread_y = tvm.thread_axis((0, y_factor), "threadIdx.y")
    thread_x = tvm.thread_axis((0, x_factor), "threadIdx.x")
    _, co, oh, ow, vc = s[conv].op.axis
    ooh, ioh = s[conv].split(oh, factor=OUTPUT_BLOCK_HEIGHT)
    oow, iow = s[conv].split(ow, factor=OUTPUT_BLOCK_WIDTH)
    s[conv].reorder(_, co, ooh, oow, vc, ioh, iow)
    coo, coi = s[conv].split(co, nparts=1)
    ooho, oohi = s[conv].split(ooh, factor=z_factor)
    oowo, oowi = s[conv].split(oow, factor=y_factor)
    vco, vci = s[conv].split(vc, factor=x_factor)
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
    s[temp_W].compute_at(s[conv_L], rc)
    if kernel.shape[3].value != 7:
        s[conv_L].unroll(ry)
        s[conv_L].unroll(rx)

    # schedule temp
    _, ci, h, w = s[temp].op.axis
    tile_and_bind3d(s, temp, ci, h, w, 1, 16, 16)

    # schedule temp_W
    _, ci, h, w = s[temp_W].op.axis
    zo, zi = s[temp_W].split(ci, 1)
    yo, yi = s[temp_W].split(h, 1)
    xo, xi = s[temp_W].split(w, 16)
    s[temp_W].reorder(zo, yo, xo, zi, yi, xi)
    s[temp_W].bind(zi, thread_z)
    s[temp_W].bind(yi, thread_y)
    s[temp_W].bind(xi, thread_x)
    s[temp_W].storage_align(s[temp_W].op.axis[2], 16, 0)

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
