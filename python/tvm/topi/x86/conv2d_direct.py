"""Conv2D operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from tvm import te
from tvm import topi
from tvm import autotvm
from tvm.topi.nn.pad import pad
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import simplify, get_const_tuple, get_const_int, tag
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..util import get_const_tuple, traverse_inline


def _fallback_schedule(cfg, wkl):
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1


def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False,
                        layout='NCHW'):
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
    wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype, layout)
    is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
    _fallback_schedule(cfg, wkl)


def conv2d_nchw_direct(Input, Filter, stride, padding, dilation, out_dtype=None):
    return conv2d_NCHW_direct(Input, Filter, stride, padding, dilation, out_dtype)


@autotvm.register_topi_compute("conv2d_nchw_direct.x86")
def conv2d_NCHW_direct(cfg, Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
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

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(cfg, te.placeholder((batch, in_channel, in_height, in_width), dtype=Input.dtype),
                            te.placeholder((num_filter, in_channel, kernel_h, kernel_w),
                                           dtype=Filter.dtype),
                            (stride_h, stride_w), padding, out_dtype)

    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw_direct")


def schedule_conv2d_nchw_direct(outs):
    """Create schedule for tensors"""
    return schedule_conv2d_NCHW_direct(outs)


@autotvm.register_topi_schedule("conv2d_nchw_direct.x86")
def schedule_conv2d_nchw_direct(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv2d_nchw_direct' in op.tag:
            conv_out = op.output(0)

            data = conv_out.op.input_tensors[0]
            kernels = conv_out.op.input_tensors[1]

            args = [s, cfg, data, kernels, conv_out, outs[0]]
            # ic, oc, kh, kw = get_const_tuple(kernel.shape)
            conv2d_nchw_direct_schedule(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s


def conv2d_nchw_direct_schedule(s, cfg, data, kernels, conv_out, last):
    _, OC, _, _ = s[conv_out].op.axis
    s[conv_out].parallel(OC)
    return s
