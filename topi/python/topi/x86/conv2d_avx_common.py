# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""Conv2D schedule on for Intel CPU"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm

from ..util import get_const_tuple
from ..nn.conv2d import _get_schedule, _get_workload
from ..nn.util import infer_pad, infer_stride
from ..nn.pad import pad
from .tensor_intrin import dot_16x1x16_int8_int8_int32
from .check_targets import check_skylake

AVXConvCommonFwd = namedtuple('AVXConvCommonFwd', ['ic_bn', 'oc_bn', 'reg_n', 'unroll_kw'])


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

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    return AVXConvCommonFwd(ic_bn, oc_bn, reg_n, False)


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
    DOPAD = (HPAD != 0 or WPAD != 0)
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
                       tvm.sum(data_vec[n, ic//sch.ic_bn, oh*HSTR+kh, ic%sch.ic_bn, ow*WSTR+kw]
                               .astype(out_dtype) *
                               kernel_vec[oc_chunk, ic//sch.ic_bn, kh, kw, ic%sch.ic_bn, oc_block]
                               .astype(out_dtype),
                               axis=[ic, kh, kw]),
                       name='conv')

    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, c // sch.oc_bn, h, w, c % sch.oc_bn]
                         .astype(out_dtype),
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


def _declaration_conv_NCHWc(wkl, sch, data, kernel):
    out_dtype = wkl.out_dtype
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size = data.shape[0]
    out_height = (wkl.height + 2 * HPAD - wkl.hkernel) // HSTR + 1
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    # pack data
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    # convolution
    oshape = (batch_size, wkl.out_filter//sch.oc_bn, out_height, out_width, sch.oc_bn)

    ic = tvm.reduce_axis((0, wkl.in_filter), name='ic')
    kh = tvm.reduce_axis((0, wkl.hkernel), name='kh')
    kw = tvm.reduce_axis((0, wkl.wkernel), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic//sch.ic_bn, oh*HSTR+kh, ow*WSTR+kw, ic%sch.ic_bn]
                               .astype(out_dtype) *
                               kernel[oc_chunk, ic//sch.ic_bn, kh, kw, ic%sch.ic_bn, oc_block],
                               axis=[ic, kh, kw]), name='conv2d_NCHWc', tag="conv2d_NCHWc")

    return conv


def _schedule_conv_NCHWc(s, wkl, sch, data, kernel, conv_out, last):
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        parallel_axis = s[A].fuse(ic_chunk, ih)
        s[A].parallel(parallel_axis)

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=sch.reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

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

    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=sch.reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(oc_chunk, oh)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


def _declaration_conv_NCHWc_int8(wkl, sch, data, kernel):
    """
    This function sets up the compute for INT8 conv 2d
    Inputs are in INT8 datatype
    Output is in INT32 datatype
    """

    out_dtype = wkl.out_dtype
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size = data.shape[0]
    out_height = (wkl.height + 2 * HPAD - wkl.hkernel) // HSTR + 1
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    # pack data
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    # convolution
    oshape = (batch_size, wkl.out_filter//sch.oc_bn, out_height, out_width, sch.oc_bn)
    kh = tvm.reduce_axis((0, wkl.hkernel), name='kh')
    kw = tvm.reduce_axis((0, wkl.wkernel), name='kw')

    # Intel performs dot product of 2 "4" Int8 values
    # Current implementation requires ic_bn to be a multiple of 4
    n_elems = 4
    assert sch.ic_bn%n_elems == 0

    ic_outer = tvm.reduce_axis((0, wkl.in_filter//(sch.ic_bn)), name='ic_outer')
    ic_f_inner = tvm.reduce_axis((0, sch.ic_bn//n_elems), name='ic_f_inner')
    ic_s_inner = tvm.reduce_axis((0, n_elems), name='ic_s_inner')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic_outer, oh*HSTR+kh, ow*WSTR+kw,
                                        ic_f_inner * n_elems +  ic_s_inner].astype(out_dtype) *
                               kernel[oc_chunk, ic_outer, kh, kw, ic_f_inner,
                                      oc_block, ic_s_inner].astype(out_dtype),
                               axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner]),
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

    # Currently INT8 operations are supported for only Skylake
    # In future the _intrin_reduce4int8 will be updated for VNNI instructions
    # In case of unsupported target, the schedule will go to the original
    # compute

    target = tvm.target.current_target(allow_none=False)
    int32_lanes = -1
    if check_skylake(target):
        int32_lanes = 16
    else:
        return s
    assert int32_lanes != -1

    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, _ = s[A].op.axis
        parallel_axis = s[A].fuse(ic_chunk, ih)
        s[A].parallel(parallel_axis)

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=sch.reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=sch.reg_n)

    # Skylake and future processors have 16 vector lanes
    assert sch.oc_bn % int32_lanes == 0

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    if sch.unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_outer, kh, ic_f_inner, kw,
                      ow_block, oc_f_inner, oc_s_inner, ic_s_inner)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_outer, kh, kw, ic_f_inner,
                      ow_block, oc_f_inner, oc_s_inner, ic_s_inner)


    pc = dot_16x1x16_int8_int8_int32()
    s[CC].tensorize(oc_s_inner, pc)
    s[CC].unroll(ow_block)
    s[CC].unroll(oc_f_inner)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=sch.reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(oc_chunk, oh)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s
