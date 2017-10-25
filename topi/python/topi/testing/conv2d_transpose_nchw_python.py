# pylint: disable=unused-variable
"""Transposed convolution in python"""
import numpy as np
import topi
from topi.nn.util import get_pad_tuple


def conv2d_transpose_nchw_python(a_np, w_np, stride, padding):
    """Transposed convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_c, in_h, in_w = a_np.shape
    out_c, _, filter_h, filter_w = w_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    # dilate stage
    dilated_a_np = topi.testing.dilate_python(a_np, [1, 1, stride_h, stride_w])
    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right
    padded_a_np = np.zeros((batch, in_c, dilated_a_np.shape[2]+bpad_top+bpad_bottom, \
        dilated_a_np.shape[3]+bpad_left+bpad_right))
    padded_a_np[:, :, bpad_top:dilated_a_np.shape[2]+bpad_top, \
        bpad_left:dilated_a_np.shape[3]+bpad_left] = dilated_a_np
    # convolution stage
    rotated_w_np = np.rot90(w_np, k=2, axes=(2, 3))
    b_np = topi.testing.conv2d_nchw_python(padded_a_np, rotated_w_np, stride=1, padding='VALID')
    return b_np
