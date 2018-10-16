# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""1x1 Conv2D schedule on for Intel CPU"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from tvm.autotvm.task import ConfigEntity

import topi

from ..nn.util import infer_pad
from ..nn.pad import pad
from .tensor_intrin import dot_16x1x16_int8_int8_int32
from .check_targets import check_skylake

AVXConv1x1Fwd = namedtuple('AVXConv1x1Fwd', ['ic_bn', 'oc_bn', 'oh_factor', 'ow_factor'])


def _get_default_schedule(wkl, simd_width):
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
                    return AVXConv1x1Fwd(ic_bn, oc_bn, oh_factor, ow_factor)

    raise ValueError("cannot decide default schedule for workload: {}".format(wkl))


def _fallback_schedule(wkl, simd_width):
    batch_size, in_channel, height, width, _ = wkl[1]
    out_channel, _, hkernel, wkernel, _ = wkl[2]
    HPAD, WPAD = wkl[4]
    HSTR, WSTR = wkl[3]
    out_height = (height + 2 * HPAD - hkernel) // HSTR + 1
    out_width = (width + 2 * WPAD - wkernel) // WSTR + 1

    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if out_channel % bn == 0:
            oc_bn = bn
            break

    ic_bn = 1
    for bn in range(oc_bn, 0, -1):
        if in_channel % bn == 0:
            ic_bn = bn
            break

    for ow_factor in range(out_width, 0, -1):
        if out_width % ow_factor == 0:
            for oh_factor in range(out_height, 0, -1):
                if out_height % oh_factor == 0 and ow_factor * oh_factor < 32:
                    cfg_dict = {"i": -1,
                                "c": None,
                                "e": [["tile_ic", "sp", [in_channel // ic_bn, ic_bn]],
                                      ["tile_oc", "sp", [out_channel // oc_bn, oc_bn]],
                                      ["tile_oh", "ot", oh_factor],
                                      ["tile_ow", "sp", [out_width // ow_factor,
                                                         ow_factor]],],
                                "t": ""}
                    return ConfigEntity.from_json_dict(cfg_dict)

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
    ic_bn, oh_factor, ow_factor = (cfg["tile_ic"].size[-1], cfg["tile_oh"].val,
                                   cfg["tile_ow"].size[-1])

    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        parallel_axis = s[A].fuse(ic_chunk, ih)
        s[A].parallel(parallel_axis)

    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[C].split(ow, factor=ow_factor)
    s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
    s[C].vectorize(oc_block)

    parallel_axis = s[C].fuse(oc_chunk, oh_outer)
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

        parallel_axis = s[O].fuse(oc_chunk, oh_outer)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


def _declaration_conv_NCHWc_int8(wkl, sch, data, kernel):
    """ Declaration for int8 conv"""
    out_dtype = wkl.out_dtype
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size = data.shape[0]
    out_height = (wkl.height + 2 * HPAD - wkl.hkernel) // HSTR + 1
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    oshape = (batch_size, wkl.out_filter//sch.oc_bn, out_height, out_width, sch.oc_bn)

    # Intel performs dot product of 2 "4" Int8 values
    n_elems = 4
    assert sch.ic_bn%n_elems == 0
    ic_outer = tvm.reduce_axis((0, wkl.in_filter//(sch.ic_bn)), name='ic_outer')
    ic_f_inner = tvm.reduce_axis((0, sch.ic_bn//n_elems), name='ic_f_inner')
    ic_s_inner = tvm.reduce_axis((0, n_elems), name='ic_s_inner')

    # Reshaping kernel as the last 2 dimensions are 1x1 (k_h x k_w)
    k_shape = kernel.shape
    kernel = topi.reshape(kernel, (k_shape[0], k_shape[1], k_shape[2], k_shape[3],
                                   k_shape[4] * k_shape[5] * k_shape[6]))

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic_outer, oh*HSTR, ow*WSTR,
                                        ic_f_inner * n_elems + ic_s_inner].astype(out_dtype) *
                               kernel[oc_chunk, ic_outer, ic_f_inner,
                                      oc_block, ic_s_inner].astype(out_dtype),
                               axis=[ic_outer, ic_f_inner, ic_s_inner]),
                       name='conv2d_NCHWc_int8',
                       tag="conv2d_NCHWc_int8")


    return conv


def _schedule_conv_NCHWc_int8(s, wkl, sch, data, kernel, conv_out, last):
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

    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        parallel_axis = s[A].fuse(ic_chunk, ih)
        s[A].parallel(parallel_axis)

    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[C].split(ow, factor=sch.ow_factor)
    s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
    s[C].vectorize(oc_block)

    parallel_axis = s[C].fuse(oc_chunk, oh_outer)
    s[CC].compute_at(s[C], parallel_axis)
    if C == O:
        s[C].parallel(parallel_axis)

    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    # Skylake and future processors have 16 vector lanes
    assert sch.oc_bn % int32_lanes == 0

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    oh_outer, oh_inner = s[CC].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=sch.ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_outer, ic_f_inner, oh_inner,
                  ow_inner, oc_f_inner, oc_s_inner, ic_s_inner)
    s[CC].fuse(oc_chunk, oh_outer)

    pc = dot_16x1x16_int8_int8_int32()
    s[CC].tensorize(oc_s_inner, pc)
    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
        ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
        s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

        parallel_axis = s[O].fuse(oc_chunk, oh_outer)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s
