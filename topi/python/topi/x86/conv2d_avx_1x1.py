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
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from ..nn.pad import pad
from ..nn.util import infer_pad, get_pad_tuple
from ..util import get_const_tuple, simplify
from .tensor_intrin import dot_16x1x16_int8_int8_int32
from .check_targets import check_skylake
from .util import get_fp32_len

def _fallback_schedule(cfg, wkl):
    simd_width = get_fp32_len()
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_height = (wkl.height + 2 * HPAD - wkl.hkernel) // HSTR + 1
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


def _schedule_conv(s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, last):
    # fetch schedule
    ic_bn, oc_bn, oh_factor, ow_factor = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                                          cfg["tile_oh"].val, cfg["tile_ow"].size[-1])

    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    HPAD, WPAD = padding
    DOPAD = (HPAD != 0 or WPAD != 0)

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec
    # schedule data
    if DOPAD:
        s[A0].compute_inline()
    batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(ic_chunk, ih)
    s[A1].parallel(parallel_axis)

    # schedule kernel pack
    oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
    if oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, oh)
    s[W].parallel(parallel_axis)

    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=oh_factor)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], oh_outer)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, _, _ = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if O0 != O:
        s[O0].compute_inline()
    batch, oc, oh, ow = s[O].op.axis

    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
    oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
    s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

    parallel_axis = s[O].fuse(oc_chunk, oh_outer)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)

    s[O].parallel(parallel_axis)

    return s


def _schedule_conv_NCHWc(s, cfg, data, conv_out, last):
    # fetch schedule
    oh_factor, ow_factor = cfg["tile_oh"].val, cfg["tile_ow"].size[-1]
    _, _, _, _, ic_bn = get_const_tuple(data.shape)

    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        parallel_axis = s[A].fuse(batch, ic_chunk, ih)
        s[A].parallel(parallel_axis)

    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

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
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
        ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
        s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

        parallel_axis = s[O].fuse(batch, oc_chunk, oh_outer)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


def _schedule_conv_NCHWc_int8(s, cfg, data, conv_out, last):
    """
    Defines the schedule for INT8 for intel machines
    Uses the Intel intrinsics to use INT8 operations
    More details - https://software.intel.com/en-us/articles/
    lower-numerical-precision-deep-learning-inference-and-training
    """
    target = tvm.target.current_target(allow_none=False)
    int32_lanes = -1
    if check_skylake(target):
        int32_lanes = 16
    else:
        return s
    assert int32_lanes != -1

    oh_factor, ow_factor = cfg["tile_oh"].val, cfg["tile_ow"].size[-1]
    _, _, _, _, ic_bn = get_const_tuple(data.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        parallel_axis = s[A].fuse(batch, ic_chunk, ih)
        s[A].parallel(parallel_axis)

    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

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
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    # Skylake and future processors have 16 vector lanes
    assert oc_bn % int32_lanes == 0

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    oh_outer, oh_inner = s[CC].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, kh, kw, ic_outer, ic_f_inner, oh_inner,
                  ow_inner, oc_f_inner, oc_s_inner, ic_s_inner)
    s[CC].fuse(oc_chunk, oh_outer)

    pc = dot_16x1x16_int8_int8_int32()
    s[CC].tensorize(oc_s_inner, pc)
    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
        ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
        s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

        parallel_axis = s[O].fuse(batch, oc_chunk, oh_outer)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


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
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # todo: padding filter to accomodate the intrinsic

    # packing the Filter to let memory access be consecutive for AVX512 intrinsic
    # Done in pre-compute stage
    packw_shape = (kernel_h, kernel_w, num_filter/16, 16*(channel/4), 4)
    PackW = tvm.compute(packw_shape, lambda a, b, c, d, e: Filter[a][b][c*16+d%16][d/16*4+e],
                        name="packed_filter")

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: tvm.sum(
            PaddedInput[nn, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            PackW[ry, rx, ff/16, (rc/4)*16+ff%16, rc%4].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2d_1x1_Output_int8", tag="conv2d_nhwc_pack_int8")
    return Output


def _schedule_conv_nhwc_pack_int8(s, cfg, data, conv_out, last):
    """
    Defines the schedule for the int8 nhwc layout. For 1x1 conv, it
    is a matrix-multiply operation by using nhwc layout. We will do
    packing of weight to make the address access be friendly to int8
    intrinsic
    """
    target = tvm.target.current_target(allow_none=False)
    int32_lanes = -1
    if check_skylake(target):
        int32_lanes = 16
    else:
        return s
    assert int32_lanes != -1

    # assertion to fail the unhandled case
    _, _, _, ic_num = get_const_tuple(data.shape)
    _, _, _, oc_num = get_const_tuple(conv_out.shape)
    assert ic_num % 4 == 0
    assert oc_num % 16 == 0

    ic_factor, oc_factor = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
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

    pc = dot_16x1x16_int8_int8_int32()
    s[C].tensorize(oc_inner, pc)

    if C != O:
        batch, last_oh, last_ow, last_oc = s[O].op.axis
        oc_chunk, oc_block = s[O].split(ochannel, 16)
        # not saw perf improvement to split oh/ow here
        s[O].vectorize(oc_block)

    return s
