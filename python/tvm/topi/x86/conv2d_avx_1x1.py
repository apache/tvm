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
# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""1x1 Conv2D schedule on for Intel CPU"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from ..nn.pad import pad
from ..nn.utils import get_pad_tuple
from ..generic import conv2d as conv2d_generic
from ..utils import get_const_tuple, simplify
from .tensor_intrin import dot_16x1x16_uint8_int8_int32
from .utils import get_simd_32bit_lanes


def _fallback_schedule(cfg, wkl):
    simd_width = get_simd_32bit_lanes()
    pt, pl, pb, pr = wkl.padt, wkl.padl, wkl.padb, wkl.padr
    HSTR, WSTR = wkl.stride_h, wkl.stride_w
    dilated_kernel_h = (wkl.kernel_h - 1) * wkl.dilation_h + 1
    dilated_kernel_w = (wkl.kernel_w - 1) * wkl.dilation_w + 1

    out_height = (wkl.height + pt + pb - dilated_kernel_h) // HSTR + 1
    out_width = (wkl.width + pl + pr - dilated_kernel_w) // WSTR + 1

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

    for ow_factor in range(out_width, 0, -1):
        if out_width % ow_factor == 0:
            for oh_factor in range(out_height, 0, -1):
                if out_height % oh_factor == 0 and ow_factor * oh_factor < 32:
                    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
                    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
                    cfg["tile_oh"] = OtherOptionEntity(oh_factor)
                    cfg["tile_ow"] = SplitEntity([out_width // ow_factor, ow_factor])
                    return
    raise ValueError("cannot decide default schedule for workload: {}".format(wkl))


def _schedule_conv_NCHWc(s, cfg, data_vec, kernel_vec, conv_out, last):
    # fetch schedule
    oh_factor, ow_factor = cfg["tile_oh"].val, cfg["tile_ow"].size[-1]
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)

    # schedule pad
    if isinstance(s[data_vec].op, tvm.te.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        s[data_vec].vectorize(ic_block)
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    oc_bn = cfg["tile_oc"].size[-1]
    if isinstance(kernel_vec.op, tvm.te.ComputeOp) and kernel_vec.name == "kernel_vec":
        # data and kernel are not pre-computed, schedule layout transform here.
        # this should only be used by x86 conv2d_nchw, which is for
        # testing purpose.
        batch, ic_chunk, ih, ic_block, iw = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)

        oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[kernel_vec].op.axis
        s[kernel_vec].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
        if oc_bn > 1:
            s[kernel_vec].vectorize(oc_block)
        parallel_axis = s[kernel_vec].fuse(oc_chunk, oh)
        s[kernel_vec].parallel(parallel_axis)

    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[C].split(ow, factor=ow_factor)
    s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
    s[C].vectorize(oc_block)

    parallel_axis = s[C].fuse(batch, oc_chunk, oh_outer)
    s[CC].compute_at(s[C], parallel_axis)
    if C == O:
        s[C].parallel(parallel_axis)

    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, _, _ = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].fuse(oc_chunk, oh_outer)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
            ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
            s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

            parallel_axis = s[O].fuse(batch, oc_chunk, oh_outer)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
            ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
            s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
            parallel_axis = s[O].fuse(batch, oc_chunk, oh_outer)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        else:
            raise ValueError("Unsupported output ndim: %s" % out_ndim)

    return s


def _schedule_conv_NCHWc_int8(s, cfg, data_vec, kernel_vec, conv_out, last):
    return conv2d_generic.schedule_conv_NCHWc_cpu_1x1_int8(
        s,
        cfg,
        data_vec,
        kernel_vec,
        conv_out,
        last,
        int32_lanes=get_simd_32bit_lanes(),
        intrin=dot_16x1x16_uint8_int8_int32(),
    )


def _declaration_conv_nhwc_pack(cfg, Input, Filter, stride, padding, dilation, out_dtype):
    # more assertion for the shapes
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, num_filter, channel = Filter.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # todo: padding filter to accommodate the intrinsic

    # packing the Filter to let memory access be consecutive for AVX512 intrinsic
    # Done in pre-compute stage
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    packw_shape = (kernel_h, kernel_w, idxd(num_filter, 16), 16 * idxd(channel, 4), 4)
    PackW = te.compute(
        packw_shape,
        lambda a, b, c, d, e: Filter[a, b, c * 16 + idxm(d, 16), idxd(d, 16) * 4 + e],
        name="packed_filter",
    )

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            PaddedInput[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * PackW[ry, rx, idxd(ff, 16), idxd(rc, 4) * 16 + idxm(ff, 16), idxm(rc, 4)].astype(
                out_dtype
            ),
            axis=[ry, rx, rc],
        ),
        name="Conv2d_1x1_Output_int8",
        tag="conv2d_nhwc_pack_int8",
    )
    return Output


def _schedule_conv_nhwc_pack_int8(s, cfg, data, conv_out, last):
    """
    Defines the schedule for the int8 nhwc layout. For 1x1 conv, it
    is a matrix-multiply operation by using nhwc layout. We will do
    packing of weight to make the address access be friendly to int8
    intrinsic
    """
    # FIXME - https://github.com/apache/tvm/issues/3598
    # pylint: disable=unreachable
    return s

    int32_lanes = 16

    # assertion to fail the unhandled case
    _, _, _, ic_num = get_const_tuple(data.shape)
    _, _, _, oc_num = get_const_tuple(conv_out.shape)
    assert ic_num % 4 == 0
    assert oc_num % 16 == 0

    ic_factor, oc_factor = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.te.ComputeOp):
        batch, ih, iw, ic = s[A].op.axis
        d_ic_chunk, d_ic_block = s[A].split(ic, factor=4)
        s[A].vectorize(d_ic_block)

    C, O = conv_out, last

    batch, oh, ow, oc = s[C].op.axis
    kh, kw, ic = s[C].op.reduce_axis
    # match the x86 intrinsic
    ic_outer, ic_inner = s[C].split(ic, factor=4)
    oc_outer, oc_inner = s[C].split(oc, factor=int32_lanes)

    ic_f_outer, ic_s_outer = s[C].split(ic_outer, factor=ic_factor)
    s[C].reorder(oc_outer, oh, ow, ic_f_outer, ic_s_outer, kh, kw, oc_inner, ic_inner)

    pc = dot_16x1x16_uint8_int8_int32()
    s[C].tensorize(oc_inner, pc)

    if C != O:
        batch, last_oh, last_ow, last_oc = s[O].op.axis
        oc_chunk, oc_block = s[O].split(ochannel, 16)
        # not saw perf improvement to split oh/ow here
        s[O].vectorize(oc_block)

    return s
