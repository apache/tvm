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
"""Conv2D schedule on for Intel CPU"""
import tvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from ..generic import conv2d as conv2d_generic
from ..utils import get_const_tuple
from .tensor_intrin import dot_16x1x16_uint8_int8_int32
from .utils import get_simd_32bit_lanes


def _fallback_schedule(cfg, wkl):
    simd_width = get_simd_32bit_lanes()
    pt, pl, pb, pr = wkl.padt, wkl.padl, wkl.padb, wkl.padr
    HSTR, WSTR = wkl.stride_h, wkl.stride_w
    dilated_kernel_w = (wkl.kernel_w - 1) * wkl.dilation_w + 1

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

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def _fallback_schedule_int8(cfg, wkl):
    pt, pl, pb, pr = wkl.padt, wkl.padl, wkl.padb, wkl.padr
    HSTR, WSTR = wkl.stride_h, wkl.stride_w
    out_width = (wkl.width + pl + pr - wkl.kernel_w) // WSTR + 1

    oc_bn = 16
    assert wkl.out_filter % oc_bn == 0

    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break
    assert wkl.in_filter % 4 == 0

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def _schedule_conv_NCHWc(s, cfg, data_vec, kernel_vec, conv_out, last):
    # fetch schedule
    reg_n, unroll_kw = cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
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

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(batch, oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(batch, oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        else:
            raise ValueError("Unsupported output ndim: %s" % out_ndim)

    return s


def _schedule_conv_NCHWc_int8(s, cfg, data_vec, kernel_vec, conv_out, last):
    return conv2d_generic.schedule_conv_NCHWc_cpu_common_int8(
        s,
        cfg,
        data_vec,
        kernel_vec,
        conv_out,
        last,
        int32_lanes=get_simd_32bit_lanes(),
        intrin=dot_16x1x16_uint8_int8_int32(),
    )
