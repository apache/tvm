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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin, no-else-return
"""Conv3D operators"""
from collections import namedtuple
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from ..utils import traverse_inline
from ..nn.utils import get_pad_tuple3d, infer_pad3d
from ..nn.pad import pad
from ..utils import get_const_tuple, simplify, get_const_int
from .utils import get_simd_32bit_lanes

Workload3D = namedtuple(
    "Workload",
    [
        "in_dtype",
        "out_dtype",
        "depth",
        "height",
        "width",
        "in_filter",
        "groups",
        "out_filter",
        "dkernel",
        "hkernel",
        "wkernel",
        "dpad",
        "hpad",
        "wpad",
        "dstride",
        "hstride",
        "wstride",
    ],
)


@autotvm.register_topi_compute("conv3d_ndhwc.x86")
def conv3d_ndhwc(cfg, data, kernel, strides, padding, dilation, groups, out_dtype):
    """3D convolution forward operator.

    Parameters
    ----------
    input : tvm.te.Tensor
        5-D input data with shapes:
        [batch, in_depth, in_height, in_width, in_channel] for NDHWC layout

    filter : tvm.te.Tensor
        5-D filter with shape [kernel_depth, kernel_height, kernel_width, in_channels, out_channels]

    strides : int or a list/tuple of three ints
        stride size, or [stride_depth, stride_height, stride_width]

    padding : int or a list/tuple of three ints
        padding size, or [pad_depth, pad_height, pad_width]

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    groups: int
        Number of groups

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_depth, out_height, out_width, out_channel] for NDHWC layout
    """
    layout = "NDHWC"
    out_dtype = data.dtype if out_dtype is None else out_dtype
    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides, strides)
    dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation, dilation)

    _create_tuning_space(cfg, data, kernel, strides, padding, dilation, groups, layout)
    if cfg.is_fallback:
        _get_default_config(cfg, data, kernel, strides, padding, groups, out_dtype, layout)
    return _conv3d_ndhwc(cfg, data, kernel, strides, padding, dilation, groups, out_dtype)


@autotvm.register_topi_compute("conv3d_ncdhw.x86")
def conv3d_ncdhw(cfg, data, kernel, strides, padding, dilation, groups, out_dtype):
    """3D convolution forward operator.

    Parameters
    ----------
    input : tvm.te.Tensor
        5-D input data with shapes:
        [batch, in_channel, in_depth, in_height, in_width] for NCDHW layout

    filter : tvm.te.Tensor
        5-D filter with shape [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]

    strides : int or a list/tuple of three ints
        stride size, or [stride_depth, stride_height, stride_width]

    padding : int or a list/tuple of three ints
        padding size, or [pad_depth, pad_height, pad_width]

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    groups: int
        Number of groups

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width] for NCDHW layout
    """
    # assert groups == 1, "conv3d_ncdhw.x86 does not support groups"
    layout = "NCDHW"
    out_dtype = data.dtype if out_dtype is None else out_dtype
    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides, strides)
    dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation, dilation)

    _create_tuning_space(cfg, data, kernel, strides, padding, dilation, groups, layout)
    if cfg.is_fallback:
        _get_default_config(cfg, data, kernel, strides, padding, groups, out_dtype, layout)
    return _conv3d_ncdhw(cfg, data, kernel, strides, padding, dilation, layout, groups, out_dtype)


@autotvm.register_topi_schedule("conv3d_ndhwc.x86")
def schedule_conv3d_ndhwc(cfg, outs):
    """TOPI schedule callback for conv3d

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv3d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv3d.
    """
    s = te.create_schedule([x.op for x in outs])

    def _traverse(op):
        if "conv3d_ndhwc" in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            kd, kh, kw, i, o = get_const_tuple(kernel.shape)
            args = [s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, outs[0]]
            _schedule_conv3d_ndhwc(*args)

    traverse_inline(s, outs[0].op, _traverse)
    return s


@autotvm.register_topi_schedule("conv3d_ncdhw.x86")
def schedule_conv3d_ncdhw(cfg, outs):
    """TOPI schedule callback for conv3d

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv3d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv3d.
    """
    s = te.create_schedule([x.op for x in outs])

    def _traverse(op):
        if "conv3d_ncdhw" in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            kd, kh, kw, i, o = get_const_tuple(kernel.shape)
            args = [s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, outs[0]]
            _schedule_conv3d_ncdhw(*args)

    traverse_inline(s, outs[0].op, _traverse)
    return s


def _conv3d_ndhwc(cfg, data, kernel, strides, padding, dilation, groups, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype

    assert isinstance(dilation, int) or len(dilation) == 3
    if isinstance(dilation, int):
        dilation_d, dilation_h, dilation_w = (dilation, dilation, dilation)
    else:
        dilation_d, dilation_h, dilation_w = dilation

    DSTR, HSTR, WSTR = strides
    batch_size, in_depth, in_height, in_width, in_channel = get_const_tuple(data.shape)
    kernel_depth, kernel_height, kernel_width, _, num_filter = get_const_tuple(kernel.shape)

    assert in_channel % groups == 0, "input channels must be a multiple of group size"
    assert num_filter % groups == 0, "number of filters must be a multiple of group size"

    dilated_kernel_d = (kernel_depth - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w)
    )

    pad_d = pad_front + pad_back
    pad_h = pad_top + pad_down
    pad_w = pad_left + pad_right

    pad_depth = in_depth + pad_d
    pad_height = in_height + pad_h
    pad_width = in_width + pad_w

    out_depth = simplify((in_depth + pad_d - dilated_kernel_d) // DSTR + 1)
    out_height = simplify((in_height + pad_h - dilated_kernel_h) // HSTR + 1)
    out_width = simplify((in_width + pad_w - dilated_kernel_w) // WSTR + 1)

    # pack data
    DOPAD = pad_d != 0 or pad_h != 0 or pad_w != 0
    if DOPAD:
        data_pad = pad(
            data,
            (0, pad_front, pad_top, pad_left, 0),
            (0, pad_back, pad_down, pad_right, 0),
            name="data_pad",
        )
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    assert groups == 1 or ic_bn <= groups
    assert groups == 1 or oc_bn <= groups
    shape = (batch_size, in_channel // ic_bn, pad_depth, pad_height, ic_bn, pad_width)
    data_vec = te.compute(
        shape, lambda n, C, d, h, c, w: data_pad[n, d, h, w, C * ic_bn + c], name="data_vec"
    )

    ci_tile = in_channel // groups // ic_bn
    if ci_tile == 0 or ci_tile * ic_bn * groups < in_channel:
        ci_tile += 1

    # pack kernel
    shape = (
        num_filter // oc_bn,
        ci_tile,
        kernel_depth,
        kernel_height,
        kernel_width,
        ic_bn,
        oc_bn,
    )
    kernel_vec = te.compute(
        shape,
        lambda CO, CI, d, h, w, ci, co: kernel[d, h, w, CI * ic_bn + ci, CO * oc_bn + co],
        name="kernel_vec",
    )

    # convolution
    oshape = (batch_size, num_filter // oc_bn, out_depth, out_height, out_width, oc_bn)
    unpack_shape = (batch_size, out_depth, out_height, out_width, num_filter)

    ic = te.reduce_axis((0, in_channel // groups), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")
    kd = te.reduce_axis((0, kernel_depth), name="kd")
    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    conv = te.compute(
        oshape,
        lambda n, oc_chunk, od, oh, ow, oc_block: te.sum(
            data_vec[
                n,
                idxdiv(
                    (oc_chunk * oc_bn + oc_block) // (num_filter // groups) * (in_channel // groups)
                    + ic,
                    ic_bn,
                ),
                od * DSTR + kd * dilation_d,
                oh * HSTR + kh * dilation_h,
                idxmod(
                    (oc_chunk * oc_bn + oc_block) // (num_filter // groups) * (in_channel // groups)
                    + ic,
                    ic_bn,
                ),
                ow * WSTR + kw * dilation_w,
            ].astype(out_dtype)
            * kernel_vec[
                oc_chunk, idxdiv(ic, ic_bn), kd, kh, kw, idxmod(ic, ic_bn), oc_block
            ].astype(out_dtype),
            axis=[kd, kh, kw, ic],
        ),
        name="conv",
    )
    conv_unpacked = te.compute(
        unpack_shape,
        lambda n, d, h, w, c: conv[n, idxdiv(c, oc_bn), d, h, w, idxmod(c, oc_bn)].astype(
            out_dtype
        ),
        name="output_unpack",
        tag="conv3d_ndhwc",
    )
    return conv_unpacked


def _conv3d_ncdhw(cfg, data, kernel, strides, padding, dilation, layout, groups, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype

    assert isinstance(dilation, int) or len(dilation) == 3
    if isinstance(dilation, int):
        dilation_d, dilation_h, dilation_w = (dilation, dilation, dilation)
    else:
        dilation_d, dilation_h, dilation_w = dilation

    DSTR, HSTR, WSTR = strides
    batch_size, in_channel, in_depth, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_depth, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    dilated_kernel_d = (kernel_depth - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w)
    )

    pad_d = pad_front + pad_back
    pad_h = pad_top + pad_down
    pad_w = pad_left + pad_right

    pad_depth = in_depth + pad_d
    pad_height = in_height + pad_h
    pad_width = in_width + pad_w

    out_depth = simplify((in_depth + pad_d - dilated_kernel_d) // DSTR + 1)
    out_height = simplify((in_height + pad_h - dilated_kernel_h) // HSTR + 1)
    out_width = simplify((in_width + pad_w - dilated_kernel_w) // WSTR + 1)

    # pack data
    DOPAD = pad_d != 0 or pad_h != 0 or pad_w != 0
    if DOPAD:
        data_pad = pad(
            data,
            (0, 0, pad_front, pad_top, pad_left),
            (0, 0, pad_back, pad_down, pad_right),
            name="data_pad",
        )
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    shape = (batch_size, in_channel // ic_bn, pad_depth, pad_height, ic_bn, pad_width)
    data_vec = te.compute(
        shape, lambda n, C, d, h, c, w: data_pad[n, C * ic_bn + c, d, h, w], name="data_vec"
    )

    ci_tile = in_channel // groups // ic_bn
    if ci_tile == 0 or ci_tile * ic_bn * groups < in_channel:
        ci_tile += 1

    # pack kernel
    shape = (
        num_filter // oc_bn,
        ci_tile,
        kernel_depth,
        kernel_height,
        kernel_width,
        ic_bn,
        oc_bn,
    )
    kernel_vec = te.compute(
        shape,
        lambda CO, CI, d, h, w, ci, co: kernel[CO * oc_bn + co, CI * ic_bn + ci, d, h, w],
        name="kernel_vec",
    )

    # convolution
    oshape = (batch_size, num_filter // oc_bn, out_depth, out_height, out_width, oc_bn)
    unpack_shape = (batch_size, num_filter, out_depth, out_height, out_width)

    ic = te.reduce_axis((0, in_channel // groups), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")
    kd = te.reduce_axis((0, kernel_depth), name="kd")
    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    conv = te.compute(
        oshape,
        lambda n, oc_chunk, od, oh, ow, oc_block: te.sum(
            data_vec[
                n,
                idxdiv(
                    (oc_chunk * oc_bn + oc_block) // (num_filter // groups) * (in_channel // groups)
                    + ic,
                    ic_bn,
                ),
                od * DSTR + kd * dilation_d,
                oh * HSTR + kh * dilation_h,
                idxmod(
                    (oc_chunk * oc_bn + oc_block) // (num_filter // groups) * (in_channel // groups)
                    + ic,
                    ic_bn,
                ),
                ow * WSTR + kw * dilation_w,
            ].astype(out_dtype)
            * kernel_vec[
                oc_chunk, idxdiv(ic, ic_bn), kd, kh, kw, idxmod(ic, ic_bn), oc_block
            ].astype(out_dtype),
            axis=[ic, kd, kh, kw],
        ),
        name="conv",
    )
    conv_unpacked = te.compute(
        unpack_shape,
        lambda n, c, d, h, w: conv[n, idxdiv(c, oc_bn), d, h, w, idxmod(c, oc_bn)].astype(
            out_dtype
        ),
        name="output_unpack",
        tag="conv3d_ncdhw",
    )
    return conv_unpacked


def _create_tuning_space(cfg, data, kernel, strides, padding, dilation, groups, layout):
    """Create schedule configuration from input arguments"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    if layout == "NDHWC":
        n, d, h, w, ic = dshape
        kd, kh, kw, _, oc = kshape
    elif layout == "NCDHW":
        n, ic, d, h, w = dshape
        oc, _, kd, kh, kw = kshape
    else:
        raise ValueError("Not support this layout {} with " "schedule template.".format(layout))

    # pad_front, pad_top, pad_left, pad_back, pad_down(bottom), pad_right
    pf, pt, pl, pb, pd, pr = get_pad_tuple3d(padding, (kd, kh, kw))
    sd, sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides, strides)
    od = (d - kd + pf + pb) // sd + 1
    oh = (h - kh + pt + pd) // sh + 1
    ow = (w - kw + pl + pr) // sw + 1

    # Create schedule config
    cfg.define_split("tile_ic", ic, num_outputs=2)
    cfg.define_split("tile_oc", oc, num_outputs=2)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 8)
    cfg.define_knob("unroll_kw", [True, False])


def _get_default_config(cfg, data, kernel, strides, padding, groups, out_dtype, layout):
    """
    Get default schedule config for the workload
    """
    if layout not in ["NDHWC", "NCDHW"]:
        raise ValueError("Layout {} is not supported".format(layout))

    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)
    wkl = _get_conv3d_workload(data, kernel, strides, padding, groups, out_dtype, layout)
    _fallback_schedule(cfg, wkl)


def _get_conv3d_workload(data, kernel, stride, padding, groups, out_dtype, data_layout="NCHW"):
    """Get the workload structure."""
    if data_layout == "NCDHW":
        _, CI, ID, IH, IW = get_const_tuple(data.shape)
        CO, CIG, KD, KH, KW = get_const_tuple(kernel.shape)
    elif data_layout == "NDHWC":
        _, ID, IH, IW, CI = get_const_tuple(data.shape)
        KD, KH, KW, CIG, CO = get_const_tuple(kernel.shape)
    else:
        raise ValueError("not support this layout {} yet".format(data_layout))

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (get_const_int(KD), get_const_int(KH), get_const_int(KW))
    )
    DPAD = pad_front + pad_back
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right
    if isinstance(stride, (tuple, list)):
        DSTR, HSTR, WSTR = stride
    else:
        DSTR, HSTR, WSTR = stride, stride, stride
    assert (data.dtype == kernel.dtype) or (
        data.dtype == "uint8" and kernel.dtype == "int8"
    ), "Do not support inputs with different data types now. ' \
        '{} vs. {}".format(
        data.dtype, kernel.dtype
    )
    return Workload3D(
        data.dtype,
        out_dtype,
        ID,
        IH,
        IW,
        CI,
        groups,
        CO,
        KD,
        KH,
        KW,
        DPAD,
        HPAD,
        WPAD,
        DSTR,
        HSTR,
        WSTR,
    )


def _fallback_schedule(cfg, wkl):
    simd_width = get_simd_32bit_lanes()
    DPAD, HPAD, WPAD = wkl.dpad, wkl.hpad, wkl.wpad
    DSTR, HSTR, WSTR = wkl.dstride, wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if wkl.out_filter % bn == 0:
            oc_bn = bn
            break

    ic_bn = 1
    for bn in range(oc_bn, 0, -1):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break

    reg_n = 1
    for n in range(7, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break
    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def _schedule_conv3d_ndhwc(s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, last):
    # fetch schedule
    ic_bn, oc_bn, reg_n, unroll_kw = (
        cfg["tile_ic"].size[-1],
        cfg["tile_oc"].size[-1],
        cfg["tile_ow"].size[-1],
        cfg["unroll_kw"].val,
    )

    # get padding size
    padding = infer_pad3d(data, data_pad, "NDHWC")
    DPAD, HPAD, WPAD = padding
    DOPAD = DPAD != 0 or HPAD != 0 or WPAD != 0

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec

    # schedule data
    if DOPAD:
        s[A0].compute_inline()
    batch, ic_chunk, idd, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(batch, ic_chunk, idd, ih)
    s[A1].parallel(parallel_axis)

    # schedule kernel pack
    oc_chunk, ic_chunk, od, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, od, oh, ic_chunk, ow, ic_block, oc_block)
    if oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, od, oh)
    s[W].parallel(parallel_axis)

    # schedule conv
    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, "global")

    _, oc_chunk, od, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, od, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, od, oh)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, od, oh, ow, oc_block = s[CC].op.axis
    kd, kh, kw, ic = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kd, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kd, kh, kw, ic_block, ow_block, oc_block)

    s[CC].fuse(oc_chunk, od, oh)
    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if O0 != O:
        s[O0].compute_inline()

    # unpacking
    batch, od, oh, ow, oc = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
    s[O].reorder(oc_chunk, od, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(batch, oc_chunk, od, oh)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)
    s[O].parallel(parallel_axis)
    return s


def _schedule_conv3d_ncdhw(s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, last):
    # fetch schedule
    ic_bn, oc_bn, reg_n, unroll_kw = (
        cfg["tile_ic"].size[-1],
        cfg["tile_oc"].size[-1],
        cfg["tile_ow"].size[-1],
        cfg["unroll_kw"].val,
    )

    # get padding size
    padding = infer_pad3d(data, data_pad, "NCDHW")
    DPAD, HPAD, WPAD = padding
    DOPAD = DPAD != 0 or HPAD != 0 or WPAD != 0

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec

    # schedule data
    if DOPAD:
        s[A0].compute_inline()
    batch, ic_chunk, idd, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(batch, ic_chunk, idd, ih)
    s[A1].parallel(parallel_axis)

    # schedule kernel pack
    oc_chunk, ic_chunk, od, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, od, oh, ic_chunk, ow, ic_block, oc_block)
    if oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, od, oh)
    s[W].parallel(parallel_axis)

    # schedule conv
    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, "global")

    _, oc_chunk, od, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, od, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, od, oh)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, od, oh, ow, oc_block = s[CC].op.axis
    ic, kd, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kd, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kd, kh, kw, ic_block, ow_block, oc_block)

    s[CC].fuse(oc_chunk, od, oh)
    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if O0 != O:
        s[O0].compute_inline()

    # unpacking
    batch, oc, od, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
    s[O].reorder(oc_chunk, od, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(batch, oc_chunk, od, oh)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)
    s[O].parallel(parallel_axis)

    return s
