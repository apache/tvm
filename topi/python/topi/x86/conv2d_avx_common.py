# pylint: disable=invalid-name,unused-variable,invalid-name
"""Conv2D schedule on for Intel CPU"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm

from ..util import get_const_tuple
from ..nn.conv2d import _get_schedule, _get_workload
from ..nn.util import infer_pad, infer_stride
from ..nn.pad import pad

AVXConvCommonFwd = namedtuple('AVXConvCommonFwd', ['ic_bn', 'oc_bn', 'reg_n', 'unroll_kw'])

def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    assert layout == 'NCHW', "only support NCHW convolution for AVX"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    # pack data
    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    shape = (batch_size, in_channel // sch.ic_bn, pad_height, sch.ic_bn, pad_width)
    data_vec = tvm.compute(shape,
                           lambda n, C, h, c, w: data_pad[n, C * sch.ic_bn + c, h, w],
                           name='data_vec')

    # pack kernel
    shape = (num_filter//sch.oc_bn, in_channel//sch.ic_bn,
             kernel_height, kernel_width, sch.ic_bn, sch.oc_bn)
    kernel_vec = tvm.compute(shape, lambda CO, CI, h, w, ci, co:
                             kernel[CO * sch.oc_bn + co, CI * sch.ic_bn + ci, h, w],
                             name='kernel_vec')

    # convolution
    oshape = (batch_size, num_filter//sch.oc_bn, out_height, out_width, sch.oc_bn)
    unpack_shape = (batch_size, num_filter, out_height, out_width)

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_vec[n, ic//sch.ic_bn, oh*HSTR+kh, ic%sch.ic_bn, ow*WSTR+kw] *
                               kernel_vec[oc_chunk, ic//sch.ic_bn, kh, kw, ic%sch.ic_bn, oc_block],
                               axis=[ic, kh, kw]),
                       name='conv')

    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, c // sch.oc_bn, h, w, c % sch.oc_bn],
                         name='output_unpack',
                         tag='conv2d_nchw')
    return unpack


def _schedule_conv(s, data, data_pad, data_vec, kernel, kernel_vec, conv_out, output, last):
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)
    wkl = _get_workload(data, kernel, stride, padding, output.dtype)
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    DOPAD = (HPAD != 0 and WPAD != 0)

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
    if sch.oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, oh)
    s[W].parallel(parallel_axis)

    # schedule conv
    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, 'global')

    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=sch.reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=sch.reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    if sch.unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].fuse(oc_chunk, oh)
    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if O0 != O:
        s[O0].compute_inline()

    batch, oc, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=sch.reg_n)
    oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
    s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(oc_chunk, oh)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)

    s[O].parallel(parallel_axis)

    return s
