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
# pylint: disable=unused-argument, redefined-builtin
"""Generic convolution schedules"""
from __future__ import absolute_import as _abs
import tvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from ..util import get_const_tuple

def fallback_schedule_cpu_common_int8(cfg, wkl, int32_lanes, num_int8_elements):
    """Fallback schedule for conv2d int8 on cpu.
    Normally the inner most pattern takes two int8/uint8 tensors
    data[num_int8_elements] and kernel[int32_lanes, num_int8_elements],
    produces a dot product int32/uint32 output[int32_lanes].

    Parameters
    ----------
    int32_lanes : int
        How many numbers of int32/uint32 will be produced using intrinsic.
        This is related to output channel.
    num_int8_elements : int
        How many numbers of input int32/uint32 will be multiplied and reduced.
        This is related to input channel.
    """
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    assert wkl.out_filter % int32_lanes == 0, \
        "wkl.out_filter=%d, int32_lanes=%d" % (wkl.out_filter, int32_lanes)
    assert wkl.in_filter % num_int8_elements == 0, \
        "wkl.in_filter=%d, num_int8_elements=%d" % (wkl.in_filter, num_int8_elements)

    oc_bn = int32_lanes
    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
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


def fallback_schedule_cpu_1x1_int8(cfg, wkl, int32_lanes, num_int8_elements):
    """Fallback schedule for 1x1 conv2d int8 on cpu.
    Normally the inner most pattern takes two int8/uint8 tensors
    data[num_int8_elements] and kernel[int32_lanes, num_int8_elements],
    produces a dot product int32/uint32 output[int32_lanes].

    Parameters
    ----------
    int32_lanes : int
        How many numbers of int32/uint32 will be produced using intrinsic.
        This is related to output channel.
    num_int8_elements : int
        How many numbers of input int32/uint32 will be multiplied and reduced.
        This is related to input channel.
    """
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_height = (wkl.height + 2 * HPAD - wkl.hkernel) // HSTR + 1
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    assert wkl.out_filter % int32_lanes == 0, \
        "wkl.out_filter=%d, int32_lanes=%d" % (wkl.out_filter, int32_lanes)
    assert wkl.in_filter % num_int8_elements == 0, \
        "wkl.in_filter=%d, num_int8_elements=%d" % (wkl.in_filter, num_int8_elements)

    oc_bn = int32_lanes
    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
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


def schedule_conv_NCHWc_cpu_common_int8(s, cfg, data, conv_out, last, int32_lanes=16, intrin=None):
    """
    Defines the schedule for INT8 for Intel and ARM machines
    Uses the Intel/ARM intrinsics to use INT8 operations
    More details - https://software.intel.com/en-us/articles/
    lower-numerical-precision-deep-learning-inference-and-training
    """
    reg_n, unroll_kw = cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
    _, _, _, _, ic_bn = get_const_tuple(data.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, _ = s[A].op.axis
        parallel_axis = s[A].fuse(batch, ic_chunk, ih)
        s[A].parallel(parallel_axis)

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)

    assert oc_bn % int32_lanes == 0
    assert ic_bn % 4 == 0  # 4 (u)int8 elements in (u)int32

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_outer, kh, ic_f_inner, kw,
                      ow_block, oc_f_inner, oc_s_inner, ic_s_inner)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_outer, kh, kw, ic_f_inner,
                      ow_block, oc_f_inner, oc_s_inner, ic_s_inner)

    if intrin is not None:
        s[CC].tensorize(oc_s_inner, intrin)
    s[CC].unroll(ow_block)
    s[CC].unroll(oc_f_inner)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(batch, oc_chunk, oh)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s

def schedule_conv_NCHWc_cpu_1x1_int8(s, cfg, data, conv_out, last, int32_lanes=16, intrin=None):
    """
    Defines the 1x1 conv schedule for INT8 for Intel and ARM machines
    Uses the Intel/ARM intrinsics to use INT8 operations
    More details - https://software.intel.com/en-us/articles/
    lower-numerical-precision-deep-learning-inference-and-training
    """
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

    assert oc_bn % int32_lanes == 0
    assert ic_bn % 4 == 0  # 4 (u)int8 elements in (u)int32

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    oh_outer, oh_inner = s[CC].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, kh, kw, ic_outer, ic_f_inner, oh_inner,
                  ow_inner, oc_f_inner, oc_s_inner, ic_s_inner)
    s[CC].fuse(oc_chunk, oh_outer)

    if intrin is not None:
        s[CC].tensorize(oc_s_inner, intrin)
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
