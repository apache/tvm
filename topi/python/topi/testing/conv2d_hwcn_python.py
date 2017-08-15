# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Convolution in python"""
import numpy as np
import scipy.signal


def conv2d_hwcn_python(a_np, w_np, stride, padding):
    """Convolution operator in HWCN layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [in_height, in_width, in_channel, batch]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [out_height, out_width, out_channel, batch]
    """
    in_height, in_width, in_channel, batch = a_np.shape
    kernel_h, kernel_w, _, num_filter = w_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == 'VALID':
        pad_h = 0
        pad_w = 0
    else: # 'SAME'
        pad_h = kernel_h - 1
        pad_w = kernel_w - 1
    pad_top = int(np.ceil(float(pad_h) / 2))
    pad_bottom = pad_h - pad_top
    pad_left = int(np.ceil(float(pad_w) / 2))
    pad_right = pad_w - pad_left
    # compute the output shape
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    # change the layout from HWCN to NCHW
    at = a_np.transpose((3, 2, 0, 1))
    wt = w_np.transpose((3, 2, 0, 1))
    bt = np.zeros((batch, out_channel, out_height, out_width))
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w))
                    apad[pad_top:-pad_bottom, pad_left:-pad_right] = at[n, c]
                else:
                    apad = at[n, c]
                out = scipy.signal.convolve2d(
                    apad, np.rot90(np.rot90(wt[f, c])), mode='valid')
                bt[n, f] += out[::stride, ::stride]
    return bt.transpose((2, 3, 1, 0))
