# pylint: disable=unused-variable
"""Transposed convolution in python"""
import numpy as np
import scipy
import topi
from topi.nn.util import get_pad_tuple


def conv2d_transpose_nchw_python(a_np, w_np, stride, padding):
    """Transposed convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [in_channel, num_filter, filter_height, filter_width]

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
    _, out_c, filter_h, filter_w = w_np.shape
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
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    b_np = np.zeros((batch, out_c, out_h, out_w))
    for n in range(batch):
        for f in range(out_c):
            for c in range(in_c):
                out = scipy.signal.convolve2d(
                    padded_a_np[n, c], w_np[c, f], mode='valid')
                b_np[n, f] += out
    return b_np
