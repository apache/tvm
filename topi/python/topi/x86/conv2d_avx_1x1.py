# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""1x1 Conv2D schedule on for Intel CPU"""
from __future__ import absolute_import as _abs
import tvm
import topi

from ..util import get_const_tuple
from ..nn.util import infer_pad
from ..nn.pad import pad
from tvm.autotvm.task import ConfigEntity


def _declaration_conv(cfg, data, kernel, strides, padding, layout, out_dtype):
    assert layout == 'NCHW', "only support NCHW convolution for AVX"

    HPAD, WPAD = padding
    HSTR, WSTR = strides

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    shape = (batch_size, in_channel // ic_bn, pad_height, pad_width, ic_bn)
    data_vec = tvm.compute(shape, lambda n, C, h, w, c: data_pad[n, C * ic_bn + c, h, w])

    shape = (num_filter // oc_bn, in_channel // ic_bn, ic_bn, oc_bn, 1, 1)
    kernel_vec = tvm.compute(shape, lambda CO, CI, ci, co, h, w:
                             kernel[CO * oc_bn + co, CI * ic_bn + ci, h, w],
                             name='kernel_vec')

    oshape = (batch_size, num_filter // oc_bn, out_height, out_width, oc_bn)
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_vec[n, ic//ic_bn, oh*HSTR, ow*WSTR, ic%ic_bn] *
                               kernel_vec[oc_chunk, ic//ic_bn, ic%ic_bn, oc_block, 0, 0],
                               axis=[ic]), name='conv')

    oshape = (batch_size, num_filter, out_height, out_width)
    unpack = tvm.compute(oshape, lambda n, oc, oh, ow:
                         conv[n, oc // oc_bn, oh, ow, oc % oc_bn],
                         tag='conv2d_nchw',
                         attrs={'workload':
                                    topi.x86.conv2d.conv_arg_to_workload(data, kernel,
                                                                         strides, padding,
                                                                         layout, out_dtype)})
    return unpack


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
    ic, = s[CC].op.reduce_axis

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


def _declaration_conv_NCHWc(cfg, data, kernel, kernel_size, strides, padding, layout,
                            out_dtype):
    HPAD, WPAD = padding
    HSTR, WSTR = strides

    n ,ic_chunk, ih, iw, ic_block = get_const_tuple(data.shape)
    ic = ic_chunk * ic_block
    kh, kw = kernel_size
    oc_chunk, _, _, oc_block, _, _ = get_const_tuple(kernel.shape)
    oc = oc_chunk * oc_block
    oh = (ih + 2 * HPAD - kh) // HSTR + 1
    ow = (iw + 2 * WPAD - kw) // WSTR + 1

    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    if ic_bn != ic_block:
        raise RuntimeError("ic_bn in config is not equal to actual data ic_block: %d vs %d."
                           % (ic_bn, ic_block))
    if oc_bn != oc_block:
        raise RuntimeError("oc_bn in config is not equal to actual kernel oc_block: %d vs %d."
                           % (oc_bn, oc_block))

    # convolution
    workload = topi.x86.conv2d.conv_NCHWc_arg_to_workload(data, kernel,
                                                          kernel_size,
                                                          strides, padding,
                                                          layout, out_dtype),
    if isinstance(cfg, ConfigEntity):
        serialized_cfg = cfg.to_json_dict()
        for k, v in serialized_cfg.items():
            if v is None:
                serialized_cfg[k] = str(v)
        attrs = {'workload': workload, 'cfg': serialized_cfg}
    else:
        attrs = {'workload': workload}
    oshape = (n, oc//oc_bn, oh, ow, oc_bn)
    ic = tvm.reduce_axis((0, ic), name='ic')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic//ic_bn, oh*HSTR, ow*WSTR, ic%ic_bn]
                               .astype(out_dtype) *
                       kernel[oc_chunk, ic // ic_bn, ic % ic_bn, oc_block, 0, 0],
                       axis=[ic]), name='conv2d_NCHWc', tag='conv2d_NCHWc',
                       attrs=attrs)
    return conv


def _schedule_conv_NCHWc(s, cfg, data, conv_out, last):
    # fetch schedule
    if isinstance(cfg, tuple):
        ic_bn, oh_factor, ow_factor = cfg[0], cfg[2], cfg[3]
    else:
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
    ic, = s[CC].op.reduce_axis

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
