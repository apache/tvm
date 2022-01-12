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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Grouped Spatial Pack Convolution (Group Conv2D) schedule on x86"""

import tvm
from tvm import autotvm
from tvm import te
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from .utils import get_simd_32bit_lanes
from ..utils import get_const_tuple
from ..nn.pad import pad
from .. import tag

from ..nn.conv2d import _get_workload as _get_conv2d_workload


def group_conv2d_nchw(data, kernel, strides, padding, dilation, groups, out_dtype):
    """Compute group_conv2d with NCHW layout"""
    return group_conv2d_nchw_spatial_pack(
        data, kernel, strides, padding, dilation, groups, out_dtype
    )


def schedule_group_conv2d_nchw(outs):
    """Compute group_conv2d with NCHW layout"""
    return schedule_group_conv2d_nchwc(outs)


def _get_default_config(
    cfg, data, kernel, strides, padding, dilation, groups, out_dtype, layout="NCHW"
):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)

    wkl = _get_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype, layout)
    _fallback_schedule(cfg, wkl)


def _fallback_schedule(cfg, wkl):
    simd_width = get_simd_32bit_lanes()
    pad_left, pad_right = wkl.padl, wkl.padr
    stride_w = wkl.stride_w
    out_width = (wkl.width + pad_left + pad_right - wkl.kernel_w) // stride_w + 1
    groups = wkl.groups
    kernels_per_group = wkl.out_filter // groups
    kernel_depth = wkl.in_filter // groups

    oc_bn = 1

    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if kernels_per_group % bn == 0:
            oc_bn = bn
            break
    if oc_bn > kernels_per_group:
        oc_bn = kernels_per_group

    ic_bn = 1
    for bn in range(oc_bn, 0, -1):
        if kernel_depth % bn == 0:
            ic_bn = bn
            break
    if ic_bn > kernel_depth:
        ic_bn = kernel_depth

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


@autotvm.register_topi_compute("group_conv2d_nchw.x86")
def group_conv2d_nchw_spatial_pack(
    cfg, data, kernel, strides, padding, dilation, groups, out_dtype="float32"
):
    """
    Compute group conv2d with NCHW layout, using GSPC algorithm.
    https://arxiv.org/abs/2006.09791
    """
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    assert isinstance(padding, int) or len(padding) == 2 or len(padding) == 4
    if isinstance(padding, int):
        pad_top, pad_left, pad_bottom, pad_right = padding, padding, padding, padding
    elif len(padding) == 2:
        hpad, wpad = padding
        pad_top, pad_bottom = hpad, hpad
        pad_left, pad_right = wpad, wpad
    else:
        pad_top, pad_left, pad_bottom, pad_right = padding

    hpad = pad_top + pad_bottom
    wpad = pad_left + pad_right

    assert isinstance(strides, int) or len(strides) == 2
    if isinstance(strides, int):
        stride_h, stride_w = strides, strides
    else:
        stride_h, stride_w = strides

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    out_channel, kernel_depth, k_height, k_width = get_const_tuple(kernel.shape)

    pad_height = in_height + pad_top + pad_bottom
    pad_width = in_width + pad_left + pad_right

    dilated_kernel_h = (k_height - 1) * dilation_h + 1
    dilated_kernel_w = (k_width - 1) * dilation_w + 1
    out_height = (in_height + pad_top + pad_bottom - dilated_kernel_h) // stride_h + 1
    out_width = (in_width + pad_left + pad_right - dilated_kernel_w) // stride_w + 1

    kernels_per_group = out_channel // groups

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", out_channel, num_outputs=2)
    cfg.define_split("tile_ow", out_width, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((batch_size, in_channel, in_height, in_width), dtype=data.dtype),
            te.placeholder(
                (out_channel, in_channel // groups, k_height, k_width), dtype=kernel.dtype
            ),
            strides,
            padding,
            dilation,
            groups,
            out_dtype,
        )

    oc_bn = cfg["tile_oc"].size[-1]
    ic_bn = cfg["tile_ic"].size[-1]

    # pack data
    DOPAD = hpad != 0 or wpad != 0
    if DOPAD:
        data_pad = pad(
            data, (0, 0, pad_top, pad_left), (0, 0, pad_bottom, pad_right), name="data_pad"
        )
    else:
        data_pad = data

    shape = (groups, batch_size, kernel_depth // ic_bn, pad_height, ic_bn, pad_width)

    data_vec = te.compute(
        shape,
        lambda g, n, C, h, c, w: data_pad[n, C * ic_bn + c + kernel_depth * g, h, w],
        name="data_vec",
    )

    # pack kernel
    shape = (
        groups,
        kernels_per_group // oc_bn,
        kernel_depth // ic_bn,
        k_height,
        k_width,
        ic_bn,
        oc_bn,
    )

    kernel_vec = te.compute(
        shape,
        lambda g, out_channel, in_channel, h, w, ci, co: kernel[
            (out_channel * oc_bn + co + g * kernels_per_group), in_channel * ic_bn + ci, h, w
        ],
        name="kernel_vec",
    )

    # convolution
    oshape = (groups, batch_size, kernels_per_group // oc_bn, out_height, out_width, oc_bn)
    unpack_shape = (batch_size, out_channel, out_height, out_width)

    ic = te.reduce_axis((0, (kernel_depth)), name="ic")
    kh = te.reduce_axis((0, k_height), name="kh")
    kw = te.reduce_axis((0, k_width), name="kw")

    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv
    conv = te.compute(
        oshape,
        lambda g, n, oc_chunk, oh, ow, oc_block: te.sum(
            data_vec[
                g,
                n,
                idxdiv(ic, ic_bn),
                oh * stride_h + kh * dilation_h,
                idxmod(ic, ic_bn),
                ow * stride_w + kw * dilation_w,
            ].astype(out_dtype)
            * kernel_vec[
                g, oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block
            ].astype(out_dtype),
            axis=[ic, kh, kw],
        ),
        name="conv",
    )

    unpack = te.compute(
        unpack_shape,
        lambda n, c, h, w: conv[
            idxdiv(c, kernels_per_group),
            n,
            idxmod(idxdiv(c, oc_bn), (kernels_per_group // oc_bn)),
            h,
            w,
            idxmod(idxmod(c, oc_bn), kernels_per_group),
        ].astype(out_dtype),
        name="output_unpack",
        tag="group_conv2d_nchw",
    )

    return unpack


@autotvm.register_topi_schedule("group_conv2d_nchw.x86")
def schedule_group_conv2d_nchwc(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if "group_conv2d_nchw" in op.tag:
            output = op.output(0)

            if "tile_ic" not in cfg:
                return
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

            args = [s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, outs[0]]
            _schedule_gspc_nchw(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


def _schedule_gspc_nchw(s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, last):
    """Schedule GSPC"""
    ic_bn, oc_bn, reg_n, unroll_kw = (
        cfg["tile_ic"].size[-1],
        cfg["tile_oc"].size[-1],
        cfg["tile_ow"].size[-1],
        cfg["unroll_kw"].val,
    )

    _, W = data, kernel_vec
    A0, A1 = data_pad, data_vec

    # schedule data
    if (
        data_pad is not None
        and isinstance(data_pad.op, tvm.te.ComputeOp)
        and "pad" in data_pad.op.tag
    ):
        s[A0].compute_inline()

    groups, batch, ic_chunk, ih, ic_block, _ = s[A1].op.axis

    parallel_axis = s[A1].fuse(batch, ic_chunk, ih)
    s[A1].parallel(parallel_axis)

    # schedule kernel pack
    groups, oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)

    if oc_bn > 1:
        s[W].vectorize(oc_block)

    parallel_axis = s[W].fuse(groups, oc_chunk, oh)
    s[W].parallel(parallel_axis)

    # schedule conv
    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, "global")

    _, _, oc_chunk, oh, ow, oc_block = s[C].op.axis

    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)

    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)

    groups, batch, oc_chunk, oh, ow, oc_block = s[CC].op.axis

    ic, kh, kw = s[CC].op.reduce_axis
    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    parallel_axis = s[CC].fuse(groups, batch, oc_chunk, oh)

    s[CC].parallel(parallel_axis)

    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_block)

    if O0 != O:
        s[O0].compute_inline()

    batch, oc, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)

    s[O].reorder(batch, oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(oc_chunk, oh)
    s[O].vectorize(oc_block)
    s[O].parallel(parallel_axis)
    return s
