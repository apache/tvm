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
# pylint: disable=invalid-name
# pylint: disable=no-value-for-parameter
"""The template for cuda group_conv2d_nchw"""
import tvm
from tvm import te
from tvm import autotvm

from .injective import schedule_injective_from_existing
from .tensor_intrin import dp4a
from ..nn.pad import pad
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..nn.utils import get_pad_tuple
from ..utils import traverse_inline, get_const_tuple, get_const_int
from .. import nn


def group_conv2d_nchw_int8(data, kernel, strides, padding, dilation, groups, out_dtype="float32"):
    """Compute group_conv2d internally using group_conv2d_nchwc layout for int8 dtype"""
    assert data.dtype in ("int8", "uint8")
    assert kernel.dtype in ("int8", "uint8")
    assert data.dtype == kernel.dtype
    packed_out = group_conv2d_NCHWc_int8(
        data, kernel, strides, padding, dilation, groups, out_dtype
    )
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_group_conv2d_nchw_int8(outs):
    """Create schedule for tensors"""
    return schedule_group_conv2d_NCHWc_int8(outs)


@autotvm.register_topi_compute("group_conv2d_nchw.cuda")
def group_conv2d_nchw(_, data, kernel, stride, padding, dilation, groups, out_dtype="float32"):
    return nn.group_conv2d_nchw(data, kernel, stride, padding, dilation, groups, out_dtype)


@autotvm.register_topi_schedule("group_conv2d_nchw.cuda")
def schedule_group_conv2d_nchw(cfg, outs):
    """TOPI schedule callback of group conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for group conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "group_conv2d_nchw":
            _schedule_group_conv2d_nchw_direct(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_group_conv2d_nchw_direct(cfg, s, conv):
    """Schedule group conv2d NCHW direct template"""
    workload = conv.op.attrs["workload"]
    groups = get_const_int(workload[6])
    num_filters = get_const_int(conv.shape[1])

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_n", n, num_outputs=4)
    cfg.define_split("tile_g", cfg.axis(groups), num_outputs=2)
    cfg.define_split("tile_f", cfg.axis(num_filters // groups), num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    g, f = s[output].split(f, nparts=groups)
    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bg, vg = cfg["tile_g"].apply(s, output, g)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bg, bf, by, bx, vn, vg, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    s[output].bind(bn, te.thread_axis("blockIdx.z"))
    s[output].bind(s[output].fuse(bg, bf), te.thread_axis("blockIdx.y"))
    s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
    s[output].bind(vn, te.thread_axis("vthread"))
    s[output].bind(vg, te.thread_axis("vthread"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))

    cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(tf, te.thread_axis("threadIdx.y"))
        tyx = s[output].fuse(ty, tx)
        s[output].bind(tyx, te.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[output], tyx)

        # number of threads
        n_tz = cfg["tile_n"].size[2]
        n_ty = cfg["tile_f"].size[2]
        n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
    else:
        s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
        s[output].bind(ty, te.thread_axis("threadIdx.y"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[output], tx)

        # number of threads
        n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
        n_ty = cfg["tile_y"].size[2]
        n_tx = cfg["tile_x"].size[2]

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, ryi = cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_ry"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        fused, tz = s[load].split(fused, factor=n_tz)
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    N, CO, OH, OW = get_const_tuple(output.shape)
    _, CI_div_groups, KH, KW = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OH * OW * CO * CI_div_groups * KH * KW)


@autotvm.register_topi_compute("group_conv2d_NCHWc_int8.cuda")
def group_conv2d_NCHWc_int8(
    cfg, data, kernel, stride, padding, dilation, groups, out_dtype="float32"
):
    """Group convolution operator for 'group_conv2d_NCHWc_int8'.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel // groups, filter_height, filter_width] or
        6-D with shape [num_filter_chunk, in_channel_chunk // groups, filter_height,
        filter_width, num_filter_block, in_channel_block]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_height, out_width, out_channel_block]
    """
    ic_block_factor = 4
    oc_block_factor = 4

    pre_computed = len(kernel.shape) == 6
    if not pre_computed:
        batch, channels, height, width = get_const_tuple(data.shape)
        out_channels, in_channels, kernel_h, kernel_w = get_const_tuple(kernel.shape)

        assert channels % groups == 0, "input channels must divide group size"
        assert out_channels % groups == 0, "output channels must divide group size"
        assert (
            channels % ic_block_factor == 0
        ), "Number of input channels per group must divide {}".format(ic_block_factor)
        assert (
            out_channels % oc_block_factor == 0
        ), "Number of output channels per group must divide {}".format(oc_block_factor)

        packed_data = te.compute(
            (batch, channels // ic_block_factor, height, width, ic_block_factor),
            lambda n, c, h, w, vc: data[n, c * ic_block_factor + vc, h, w],
            name="packed_data",
        )
        packed_kernel = te.compute(
            (
                out_channels // oc_block_factor,
                in_channels // ic_block_factor,
                kernel_h,
                kernel_w,
                oc_block_factor,
                ic_block_factor,
            ),
            lambda oc_chunk, ic_chunk, kh, kw, oc_block, ic_block: kernel[
                oc_chunk * oc_block_factor + oc_block, ic_chunk * ic_block_factor + ic_block, kh, kw
            ],
            name="packed_kernel",
        )
    else:
        packed_data = data
        packed_kernel = kernel

    batch, ic_chunk, in_height, in_width, _ = get_const_tuple(packed_data.shape)
    oc_chunk, _, kernel_h, kernel_w, oc_block, ic_block = get_const_tuple(packed_kernel.shape)

    # TODO(kumasento): these assertions ensure that the number of groups
    # should be smaller or equal to the number of blocks, so that each
    # group will have at least one block.
    # Shall we pad the channels to avoid raising assertions?
    assert (
        groups <= oc_chunk
    ), "Number of groups {} should be less than " "output channel chunk size {}".format(
        groups, oc_chunk
    )
    assert (
        groups <= ic_chunk
    ), "Number of groups {} should be less than " "input channel chunk size {}".format(
        groups, ic_chunk
    )

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    # pad the input data
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    pad_data = pad(packed_data, pad_before, pad_after, name="pad_data")

    # compute the output shape
    out_height = (in_height - (kernel_h - 1) * dilation_h - 1 + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - (kernel_w - 1) * dilation_w - 1 + pad_left + pad_right) // stride_w + 1

    oshape = (batch, oc_chunk, out_height, out_width, oc_block)

    icc = te.reduce_axis((0, ic_chunk // groups), name="ic_chunk")
    icb = te.reduce_axis((0, ic_block_factor), name="ic_block")
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")

    # NOTE(kumasento): explanation of this snippet -
    # oc_chunk//groups and ic_chunk//groups give you the number of blocks,
    # i.e., chunk, per group.
    # occ is the ID of the output channel block, so that occ//(oc_chunk//groups)
    # produces the ID of the group.
    # Multiplying that result with ic_chunk//groups resulting in the ID
    # of the beginning block of the corresponding input group.
    # Adding the block offset (icc) will give you the exact block ID.
    #
    # Compared with a normal convolution, group convolution only sums
    # input channels from the group that an output channel resides in.
    conv = te.compute(
        oshape,
        lambda n, occ, oh, ow, ocb: te.sum(
            pad_data[
                n,
                occ // (oc_chunk // groups) * (ic_chunk // groups) + icc,
                oh * stride_h + kh * dilation_h,
                ow * stride_w + kw * dilation_w,
                icb,
            ].astype("int32")
            * packed_kernel[occ, icc, kh, kw, ocb, icb].astype("int32"),
            axis=[icc, kh, kw, icb],
        ),
    )

    # Type conversion
    output = te.compute(
        oshape, lambda *index: conv(*index).astype(out_dtype), tag="group_conv2d_NCHWc_int8"
    )

    num_flop = (
        batch
        * oc_chunk
        * oc_block
        * out_height
        * out_width
        * ic_chunk
        * ic_block
        * kernel_h
        * kernel_w
        * 2
        // groups
    )
    cfg.add_flop(num_flop)

    return output


@autotvm.register_topi_schedule("group_conv2d_NCHWc_int8.cuda")
def schedule_group_conv2d_NCHWc_int8(cfg, outs):
    """TOPI schedule callback of group conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for group conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "group_conv2d_NCHWc_int8":
            _schedule_group_conv2d_NCHWc_int8(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_group_conv2d_NCHWc_int8(cfg, s, output):
    """Schedule group conv2d int8 NCHWc template"""
    workload = output.op.attrs["workload"]
    groups = get_const_int(workload[6])

    conv = output.op.input_tensors[0]
    packed_data, packed_kernel = conv.op.input_tensors

    if isinstance(packed_data.op, tvm.te.ComputeOp) and "pad" in packed_data.op.tag:
        pad_data = packed_data
        packed_data = pad_data.op.input_tensors[0]
    else:
        pad_data = packed_data

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # skip this part during tuning to make records accurate
        # this part will be pre-computed during NNVM's pre-compute optimization pass
        s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
        s[packed_kernel].pragma(s[packed_kernel].op.axis[0], "debug_skip_region")
    else:
        if isinstance(packed_kernel.op, tvm.te.ComputeOp) and packed_kernel.name == "packed_kernel":
            # data and kernel are not pre-computed, schedule layout transform here
            schedule_injective_from_existing(s, packed_data)
            schedule_injective_from_existing(s, packed_kernel)

    if pad_data != packed_data:
        s[pad_data].compute_inline()

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [conv])
    WW = s.cache_read(packed_kernel, "shared", [conv])

    s[conv].set_scope("local")

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    oc_chunk = get_const_int(output.shape[1])
    # tile and bind spatial axes
    if len(s[output].op.axis) == 5:
        n, f, y, x, c = s[output].op.axis
    else:
        # For task extraction of auto-tuning, the expected output is 4D.  Since auto-tuning tasks
        # are created from scratch, therefore the real auto-tuning will still happen on 5D output.
        n, f, y, x = s[output].op.axis

    cfg.define_split("tile_n", n, num_outputs=4)
    cfg.define_split("tile_g", cfg.axis(groups), num_outputs=2)
    cfg.define_split("tile_f", cfg.axis(oc_chunk // groups), num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    g, f = s[output].split(f, nparts=groups)
    s[output].bind(n, te.thread_axis("blockIdx.z"))
    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bg, vg = cfg["tile_g"].apply(s, output, g)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bg, bf, by, bx, vn, vg, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    s[output].bind(bn, te.thread_axis("blockIdx.z"))
    s[output].bind(s[output].fuse(bg, bf), te.thread_axis("blockIdx.y"))
    s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
    s[output].bind(vn, te.thread_axis("vthread"))
    s[output].bind(vg, te.thread_axis("vthread"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(tf, te.thread_axis("threadIdx.y"))
        tyx = s[output].fuse(ty, tx)
        s[output].bind(tyx, te.thread_axis("threadIdx.x"))
        s[conv].compute_at(s[output], tyx)

        # number of threads
        n_tz = cfg["tile_n"].size[2]
        n_ty = cfg["tile_f"].size[2]
        n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
    else:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
        s[output].bind(ty, te.thread_axis("threadIdx.y"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))
        s[conv].compute_at(s[output], tx)

        # number of threads
        n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
        n_ty = cfg["tile_y"].size[2]
        n_tx = cfg["tile_x"].size[2]

    # tile and bind reduction axes
    n, f, y, x, c = s[conv].op.axis
    rc, ry, rx, rc_block = s[conv].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=2)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=2)
    rco, rci = cfg["tile_rc"].apply(s, conv, rc)
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x, c, rc_block)
    _, rc_block = s[conv].split(rc_block, factor=4)
    target = tvm.target.Target.current(allow_none=False)
    do_tensorize = "+dotprod" in target.mattr or target.supports_integer_dot_product
    if do_tensorize:
        dtypes = (pad_data.dtype, packed_kernel.dtype)
        s[conv].tensorize(rc_block, dp4a("shared", "shared", "local", dtypes))

    s[AA].compute_at(s[conv], rxo)
    s[WW].compute_at(s[conv], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        c = s[load].op.axis[-1]
        c_outer, c = s[load].split(c, factor=4)
        s[load].vectorize(c)
        fused = s[load].op.axis[:-1] + [c_outer]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        fused, tz = s[load].split(fused, factor=n_tz)
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # double buffer
    cfg.define_knob("AA_double_buffer", [0, 1])
    cfg.define_knob("WW_double_buffer", [0, 1])
    if cfg["AA_double_buffer"].val:
        s[AA].double_buffer()
    if cfg["WW_double_buffer"].val:
        s[WW].double_buffer()

    # unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", False)

    return s
