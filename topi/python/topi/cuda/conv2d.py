# pylint: disable=invalid-name
"""Compute definition for conv2d with cuda backend"""
import tvm
from tvm import autotvm
from tvm.contrib import cudnn

from .. import nn, generic
from ..util import get_const_int, get_const_tuple, traverse_inline

from .conv2d_direct import schedule_direct_cuda
from .conv2d_winograd import winograd_cuda, schedule_winograd_cuda
from .conv2d_int8 import conv2d_NCHWc_int8, schedule_conv2d_NCHWc_int8


@autotvm.register_topi_compute(nn.conv2d, ['cuda', 'gpu'], ['direct', 'winograd', 'int8'])
def conv2d_cuda(cfg, data, kernel, strides, padding, layout='NCHW', out_dtype='float32'):
    """Conv2D operator for cuda backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
        5-D with shape [batch, ic_chunk, in_height, in_width, ic_block]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        6-D with shape [num_filter_chunk, in_channel_chunk, filter_height,
        filter_width, num_filter_block, in_channel_block]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    target = tvm.target.current_target()

    if "cudnn" in target.libs:
        if layout == 'NCHW':
            tensor_format = 0 # CUDNN_TENSOR_NCHW
            N, _, H, W = get_const_tuple(data.shape)
        elif layout == 'NHWC':
            tensor_format = 1 # CUDNN_TENSOR_NHWC
            N, H, W, _ = get_const_tuple(data.shape)
        else:
            raise ValueError("Unsupported layout %s in cudnn" % layout)
        CO, CI, KH, KW = get_const_tuple(kernel.shape)

        # handle dilation
        stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding

        OH = (H + 2 * pad_h - KH) // stride_h + 1
        OW = (W + 2 * pad_w - KW) // stride_w + 1
        cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)

        dilation_h = dilation_w = 1
        kernel_before_dilation = kernel
        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            kernel_before_dilation = kernel.op.input_tensors[0]
            if layout == 'NCHW':
                dilation_h = (get_const_int(kernel.shape[2]) +
                              get_const_int(kernel_before_dilation.shape[2]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[2])
                dilation_w = (get_const_int(kernel.shape[3]) +
                              get_const_int(kernel_before_dilation.shape[3]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[2])
            elif layout == 'NHWC':
                dilation_h = (get_const_int(kernel.shape[1]) +
                              get_const_int(kernel_before_dilation.shape[1]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[1])
                dilation_w = (get_const_int(kernel.shape[2]) +
                              get_const_int(kernel_before_dilation.shape[2]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[2])

        return cudnn.conv2d_forward(data,
                                    kernel_before_dilation,
                                    stride_h,
                                    stride_w,
                                    pad_h,
                                    pad_w,
                                    dilation_h,
                                    dilation_w,
                                    conv_mode=1,
                                    tensor_format=tensor_format,
                                    algo=-1)  # let CUDNN choose the best algo

    if cfg.template_key == 'winograd':
        return winograd_cuda(cfg, data, kernel, strides, padding, layout, out_dtype,
                             pre_computed=False)
    if cfg.template_key == 'int8':
        return conv2d_NCHWc_int8(cfg, data, kernel, strides, padding, layout, out_dtype,
                                 pre_computed=False)

    if layout == 'NCHW':
        return nn.conv2d_nchw(data, kernel, strides, padding, out_dtype)
    elif layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, ["cuda", "gpu"],
                                ["direct", 'winograd', "int8"])
def schedule_conv2d_nchw_cuda(cfg, outs):
    """TOPI schedule callback of conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    target = tvm.target.current_target()
    if 'cudnn' in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv2d_nchw':
            schedule_direct_cuda(cfg, s, op.output(0))
        if op.tag == 'conv2d_nchw_winograd':
            schedule_winograd_cuda(cfg, s, op.output(0), pre_computed=False)
        if op.tag == "conv2d_NCHWc_int8":
            schedule_conv2d_NCHWc_int8(cfg, s, op.output(0), pre_computed=False)

    traverse_inline(s, outs[0].op, _callback)
    return s
