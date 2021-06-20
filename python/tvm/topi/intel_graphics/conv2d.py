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
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from .. import nn
from .. import utils
from ..utils import simplify, get_const_tuple, traverse_inline


def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False):
    if is_depthwise:
        raise RuntimeError("Depthwise not supported for intel graphics.")

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


def _create_schedule_template(cfg, dshape, kshape, strides, padding, dilation):
    """Create schedule configuration from input arguments"""
    n, ic, h, w = dshape
    oc, _, kh, kw = kshape

    pt, pl, pb, pr = nn.get_pad_tuple(padding, (kh, kw))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (h - kh + pt + pb) // sh + 1
    ow = (w - kw + pl + pr) // sw + 1
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
    """tile and bind 3d"""
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].reorder(zo, yo, xo, zi, yi, xi)

    thread_z = te.thread_axis((0, z_factor), "threadIdx.z")
    thread_y = te.thread_axis((0, y_factor), "threadIdx.y")
    thread_x = te.thread_axis((0, x_factor), "threadIdx.x")
    s[tensor].bind(zo, te.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, thread_z)
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, thread_y)
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, thread_x)
    return xi, thread_z, thread_y, thread_x


def _pack_data(data, kernel, ic_bn, oc_bn):
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)

    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    data = te.compute(
        (n, ic_chunk, ih, iw, ic_bn),
        lambda bs, c, h, w, vc: data[bs, c * ic_bn + vc, h, w],
        name="data_vec",
    )

    kernel = te.compute(
        (oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn),
        lambda occ, icc, k_h, k_w, icb, ocb: kernel[occ * oc_bn + ocb, icc * ic_bn + icb, k_h, k_w],
        name="kernel_vec",
    )

    return data, kernel


@autotvm.register_topi_compute("conv2d_NCHWc.intel_graphics")
def conv2d_NCHWc(
    cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype="float32"
):
    """Conv2D operator for Intel Graphics backend.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        5-D with shape [num_filter, in_channel, filter_height, filter_width, nnum_filter_vec]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if len(data.shape) == 5:
        batch, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, _, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        batch, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (kernel_height, kernel_width)
    )
    assert (dh, dw) == (1, 1), "Does not support dilation"
    if isinstance(strides, (tuple, list)):
        stride_h, stride_w = strides
    else:
        stride_h, stride_w = strides, strides

    data_shape = (batch, in_channel, ih, iw)
    kernel_shape = (num_filter, in_channel, kernel_height, kernel_width)
    _create_schedule_template(cfg, data_shape, kernel_shape, strides, padding, dilation)

    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((batch, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder(
                (num_filter, in_channel, kernel_height, kernel_width), dtype=kernel.dtype
            ),
            strides,
            padding,
            out_dtype,
        )

    ic_bn = cfg["tile_ic"].val if hasattr(cfg["tile_ic"], "val") else cfg["tile_ic"].size[-1]
    oc_bn = cfg["tile_oc"].val if hasattr(cfg["tile_oc"], "val") else cfg["tile_oc"].size[-1]

    # Pack data if raw 4-D data is provided.
    if len(data.shape) == 4:
        data, kernel = _pack_data(data, kernel, ic_bn, oc_bn)

    out_channel = num_filter
    out_height = simplify((ih - kernel_height + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((iw - kernel_width + pad_left + pad_right) // stride_w + 1)
    oshape = (batch, out_channel // oc_bn, out_height, out_width, oc_bn)

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_height), name="ry")
    rx = te.reduce_axis((0, kernel_width), name="rx")

    block_h = cfg["block_oh"].val
    block_w = cfg["block_ow"].val

    c_h = out_height
    c_w = out_width

    if out_height % block_h != 0:
        c_h = (out_height // block_h + 1) * block_h

    if out_width % block_w != 0:
        c_w = (out_width // block_w + 1) * block_w

    cshape = (batch, out_channel // oc_bn, c_h, c_w, oc_bn)

    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down + c_h - out_height, pad_right + c_w - out_width, 0]
    DOPAD = (
        pad_top != 0
        or pad_left != 0
        or pad_down + c_h - out_height != 0
        or pad_right + c_w - out_width != 0
    )
    DOUNPACK = c_h - out_height != 0 or c_w - out_width != 0
    if DOPAD:
        temp = nn.pad(data, pad_before, pad_after, name="pad_temp")
    else:
        temp = data

    conv = te.compute(
        cshape,
        lambda nn, ff, yy, xx, ff_v: te.sum(
            temp[nn, rc // ic_bn, yy * stride_h + ry, xx * stride_w + rx, rc % ic_bn].astype(
                out_dtype
            )
            * kernel[ff, rc // ic_bn, ry, rx, rc % ic_bn, ff_v].astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        tag="conv2d_NCHWc",
        name="conv2d_NCHWc",
    )

    if DOUNPACK:
        output = te.compute(
            oshape,
            lambda nn, ff, yy, xx, ff_v: conv[nn][ff][yy][xx][ff_v],
            name="output_unpack",
            tag="conv2d_NCHWc_unpack",
        )
    else:
        output = conv

    return output


@autotvm.register_topi_schedule("conv2d_NCHWc.intel_graphics")
def schedule_conv2d_NCHWc(cfg, outs):
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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if "conv2d_NCHWc" in op.tag:
            _schedule_cl_spatialpack_NCHWc(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)

    return s


def _schedule_cl_spatialpack_NCHWc(cfg, s, op):
    output = op.output(0)
    if op.name == "conv2d_NCHWc":
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
    else:  # conv2d_NCHWc_unpack
        conv = op.input_tensors[0]
        temp = s[conv].op.input_tensors[0]
        kernel = s[conv].op.input_tensors[1]
        temp_W = s.cache_read(temp, "warp", [conv])
        conv_L = s.cache_write(conv, "local")
        SCHEDULE_OUTPUT = True
    kernel_L = s.cache_read(kernel, "local", [conv_L])

    if temp.name == "pad_temp":
        data = temp.op.input_tensors[0]
        # TODO(@Laurawly): Do we need to schedule pad op here?
    else:
        data = temp

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # only in autotuning, input data of conv2d_NCHWc will be 4-D.
        # skip this part during tuning to make records accurate.
        # this part will be folded during Relay fold_constant pass.
        s[data].pragma(s[data].op.axis[0], "debug_skip_region")
        s[kernel].pragma(s[kernel].op.axis[0], "debug_skip_region")
    elif isinstance(kernel.op, tvm.te.ComputeOp) and kernel.name == "kernel_vec":
        # data and kernel are not pre-computed, schedule layout transform here.
        # TODO(@Laurawly): Add schedule for data and kernel pack
        pass

    OUTPUT_BLOCK_HEIGHT = cfg["block_oh"].val
    OUTPUT_BLOCK_WIDTH = cfg["block_ow"].val

    # schedule conv
    z_factor = 1
    y_factor = 1
    x_factor = 16
    thread_z = te.thread_axis((0, z_factor), "threadIdx.z")
    thread_y = te.thread_axis((0, y_factor), "threadIdx.y")
    thread_x = te.thread_axis((0, x_factor), "threadIdx.x")
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
    s[conv].bind(ooho, te.thread_axis("blockIdx.z"))
    s[conv].bind(oowo, te.thread_axis("blockIdx.y"))
    s[conv].bind(coi, te.thread_axis("blockIdx.x"))

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


def conv2d_nchw(data, kernel, stride, padding, dilation, out_dtype="float32"):
    """Conv2D operator for Intel Graphics backend.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]
    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert data.shape[0].value == 1, "only support batch size=1 convolution on intel gpu"
    assert data.dtype == kernel.dtype, "Do not support inputs with different data types now."

    return _decl_cl_spatialpack(data, kernel, stride, padding, out_dtype)


def schedule_conv2d_nchw(outs):
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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if "conv2d" in op.tag:
            _schedule_cl_spatialpack(s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_cl_spatialpack(data, kernel, stride, padding, out_dtype="float16"):
    batch, in_channel, in_height, in_width = [utils.get_const_int(x) for x in data.shape]
    num_filter, channel, kernel_h, kernel_w = [utils.get_const_int(x) for x in kernel.shape]
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(padding, (kernel_h, kernel_w))

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride

    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    oshape = (batch, out_channel, out_height, out_width)

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

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
    attrs = {"block_h": block_h, "block_w": block_w}
    c_h = out_height
    c_w = out_width

    if out_height % block_h != 0:
        c_h = (out_height // block_h + 1) * block_h

    if out_width % block_w != 0:
        c_w = (out_width // block_w + 1) * block_w

    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down + c_h - block_h, pad_right + c_w - block_w]
    temp = nn.pad(data, pad_before, pad_after, name="pad_temp")

    nv = 16
    if num_filter % nv != 0:
        num_filter = (num_filter // nv + 1) * nv
        out_channel = num_filter

    cshape = (batch, out_channel // nv, c_h, c_w, nv)
    kvshape = (num_filter // nv, channel, kernel_h, kernel_w, nv)

    kernel_vec = te.compute(
        kvshape, lambda co, ci, kh, kw, vc: kernel[co * nv + vc][ci][kh][kw], name="kernel_vec"
    )

    conv = te.compute(
        cshape,
        lambda nn, ff, yy, xx, vc: te.sum(
            temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx].astype(out_dtype)
            * kernel_vec[ff, rc, ry, rx, vc].astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        name="conv",
        attrs=attrs,
    )

    output = te.compute(
        oshape,
        lambda nn, ff, yy, xx: conv[nn][ff // nv][yy][xx][ff % nv],
        name="output_unpack",
        tag="conv2d",
    )

    return output


def _schedule_cl_spatialpack(s, op):
    output = op.output(0)
    _, _, out_height, out_width = [utils.get_const_int(x) for x in output.shape]

    conv = op.input_tensors[0]
    temp = s[conv].op.input_tensors[0]
    kernel_vec = s[conv].op.input_tensors[1]
    kernel = s[kernel_vec].op.input_tensors[0]
    temp_W = s.cache_read(temp, "shared", [conv])
    conv_L = s.cache_write(conv, "local")

    kernel_L = s.cache_read(kernel_vec, "local", [conv_L])
    _, in_channel, temp_h, temp_w = [utils.get_const_int(x) for x in temp.shape]

    attrs = s[conv].op.attrs
    OUTPUT_BLOCK_HEIGHT = attrs["block_h"]
    OUTPUT_BLOCK_WIDTH = attrs["block_w"]

    # schedule conv
    z_factor = 1
    y_factor = 1
    x_factor = 16
    thread_z = te.thread_axis((0, z_factor), "threadIdx.z")
    thread_y = te.thread_axis((0, y_factor), "threadIdx.y")
    thread_x = te.thread_axis((0, x_factor), "threadIdx.x")
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
    s[conv].bind(ooho, te.thread_axis("blockIdx.z"))
    s[conv].bind(oowo, te.thread_axis("blockIdx.y"))
    s[conv].bind(coi, te.thread_axis("blockIdx.x"))

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
    if OUTPUT_BLOCK_HEIGHT == 2 and OUTPUT_BLOCK_WIDTH == 14:
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
