# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Depthwise Conv2D schedule on x86"""
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from .. import generic, tag
from ..nn.pad import pad
from ..util import get_const_tuple
from ..nn.util import get_pad_tuple
from ..nn.depthwise_conv2d import depthwise_conv2d_NCHWc, _get_workload

from .util import get_fp32_len

def _fallback_schedule(cfg, wkl):
    """
    Get default schedule for the workload
    Parameters
    ----------
    cfg : tvm.autotvm.task.space.FallbackConfigEntity
        Fallback config to be updated
    wkl : topi.nn.depthwise_conv2d.Workload
        Convolution workload
    """
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

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])


@autotvm.register_topi_compute(depthwise_conv2d_NCHWc, 'cpu', 'direct')
def _depthwise_conv2d_NCHWc_cpu(cfg, data, kernel, strides, padding,
                                layout, out_layout, out_dtype=None):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    batch, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(data.shape)
    out_channel_chunk, filter_height, filter_width, out_channel_block \
        = get_const_tuple(kernel.shape)

    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HSTR, WSTR = strides
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, (filter_height, filter_width))

    in_channel = in_channel_chunk * in_channel_block
    out_channel = out_channel_chunk * out_channel_block
    channel_multiplier = out_channel // in_channel

    out_height = (in_height - filter_height + pad_top + pad_down) // HSTR + 1
    out_width = (in_width - filter_width + pad_left + pad_right) // WSTR + 1

    # get workload and related schedule config
    wkl = _get_workload(tvm.placeholder((batch, in_channel, in_height, in_width), dtype=data.dtype),
                        tvm.placeholder((out_channel, in_channel, filter_height, filter_width),
                                        dtype=kernel.dtype),
                        strides, padding, out_dtype)
    if cfg.is_fallback:
        _fallback_schedule(cfg, wkl)

    # padding stage
    DOPAD = (pad_top != 0 or pad_left != 0 or pad_down != 0 or pad_right != 0)
    if DOPAD:
        pad_before = [0, 0, pad_top, pad_left, 0]
        pad_after = [0, 0, pad_down, pad_right, 0]
        data_pad = pad(data, pad_before, pad_after, name="PaddedInput")
    else:
        data_pad = data

    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    Output = tvm.compute(
        (batch, out_channel_chunk, out_height, out_width, out_channel_block),
        lambda b, oco, i, j, oci: tvm.sum(
            (data_pad[b, (oco * out_channel_block + oci) // channel_multiplier // in_channel_block,
                      i*HSTR+di, j*WSTR+dj,
                      ((oco * out_channel_block + oci) // channel_multiplier) % in_channel_block]
             .astype(out_dtype) *
             kernel[oco, di, dj, oci].astype(out_dtype)),
            axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_NCHWc")
    return Output


@autotvm.register_topi_schedule(generic.schedule_depthwise_conv2d_NCHWc, 'cpu', ['direct'])
def schedule_depthwise_conv2d_NCHWc(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        if 'depthwise_conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            data = conv_out.op.input_tensors[0]
            input = data
            kernel = conv_out.op.input_tensors[1]
            _schedule_depthwise_conv2d_NCHWc_impl(s, cfg, input, kernel, conv_out, outs[0])
        scheduled_ops.append(op)
    traverse(outs[0].op)
    return s

def _schedule_depthwise_conv2d_NCHWc_impl(s, cfg, data, kernel, conv_out, output):
    tile_ow = cfg["tile_ow"].size[-1]
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        p = s[A].fuse(ic_chunk, ih)
        s[A].parallel(p)

    C, O = conv_out, output
    CC = s.cache_write(C, 'global')

    _, ic_chunk, oh, ow, ic_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=tile_ow)
    s[C].reorder(ic_chunk, oh, ow_chunk, ow_block, ic_block)
    s[C].vectorize(ic_block)
    parallel_axis = s[C].fuse(ic_chunk, oh)
    s[C].parallel(parallel_axis)
    s[C].unroll(ow_block)
    s[CC].compute_at(s[C], ow_chunk)

    _, ic_chunk, oh, ow, ic_block = s[CC].op.axis
    kh, kw = s[CC].op.reduce_axis
    ow_chunk, ow_block = s[CC].split(ow, factor=tile_ow)
    s[CC].reorder(ic_chunk, oh, kh, kw, ow_block, ic_block)
    s[CC].vectorize(ic_block)
    s[CC].unroll(ow_block)
    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=tile_ow)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(oc_chunk, oh)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)
    return s
