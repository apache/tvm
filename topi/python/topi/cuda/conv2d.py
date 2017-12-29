# pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments, too-many-branches, line-too-long
"""Compute definition for conv2d with cuda backend"""
import tvm
from tvm.contrib import cudnn
import topi
from ..nn.conv2d import conv2d

@conv2d.register("cuda")
def conv2d_cuda(data, kernel, stride, padding, layout='NCHW', out_dtype='float32'):
    """Conv2D operator for cuda backend.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert isinstance(stride, int) or len(stride) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    target = tvm.target.current_target()
    if "cudnn" in target.libs:
        assert layout != 'HWCN', "HWCN layout not supported with CUDNN."
        tensor_format = 0 # CUDNN_TENSOR_NCHW
        if layout == 'NHWC':
            tensor_format = 1 # CUDNN_TENSOR_NHWC
        return cudnn.conv2d_forward(data,
                                    kernel,
                                    stride_h,
                                    stride_w,
                                    pad_h,
                                    pad_w,
                                    1,  # dilation_h
                                    1,  # dilation_w
                                    conv_mode=1,
                                    tensor_format=tensor_format,
                                    algo=-1) # let CUDNN choose the best algo
    elif layout == 'NCHW':
        return topi.nn.conv2d_nchw(data, kernel, stride, padding, out_dtype)
    elif layout == 'HWCN':
        return topi.nn.conv2d_hwcn(data, kernel, stride, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))
