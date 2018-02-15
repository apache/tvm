# pylint: disable=invalid-name, unused-variable, line-too-long
"""NN operator common utilities"""
from __future__ import absolute_import
import tvm
from ..util import get_const_int
from ..util import simplify

def infer_pad(data, data_pad):
    """Infer the padding from stages in reverse.

    Parameters
    ----------
    data : Tensor
        data stage.

    data_pad : Tensor
        pad stage.

    Returns
    -------
    hpad : int
        padding size on height
    wpad : int
        padding size on width
    """
    if data_pad is None:
        return 0, 0
    _, _, IH, IW = data.shape
    _, _, TH, TW = data_pad.shape
    hpad = (TH - IH) // 2
    wpad = (TW - IW) // 2
    return get_const_int(hpad), get_const_int(wpad)

def infer_stride(data, kernel, out):
    """Infer the stride from stages in reverse.

    Parameters
    ----------
    data : Tensor
        data stage.

    kernel : Tensor
        kernel stage.

    out : Tensor
        output stage.

    Returns
    -------
    hstride : int
        stride size on height
    wstride : int
        stride size on width
    """
    _, _, IH, IW = data.shape
    _, _, KH, KW = kernel.shape
    _, _, OH, OW = out.shape
    hstride = (IH - KH) // (OH - 1)
    wstride = (IW - KW) // (OW - 1)
    return get_const_int(hstride), get_const_int(wstride)


def get_pad_tuple(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_top : int
        Padding size on top

    pad_left : int
        Padding size on left

    pad_down : int
        Padding size on down.

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        pad_h = padding[0] * 2
        pad_w = padding[1] * 2
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left

def get_scheme_padding(padding_scheme, padding, layout, kernel_shape, data_shape, stride):
    """ Padding calculation based on the scheme.
        Compatible to Tensorflow 'SAME' or 'VALID' options
    """

    pad_h = padding[0]
    pad_w = padding[1]

    if padding_scheme == 1:
        kernel_h = kernel_shape[0]
        kernel_w = kernel_shape[1]
        if layout == "NCHW":
            in_height = data_shape[2]
            in_width = data_shape[3]
        else: #// NHWC
            in_height = data_shape[1]
            in_width = data_shape[2]

        pad_h = tvm.select(tvm.all((in_height % stride[0]) == 0), simplify(kernel_h - stride[0]), simplify(kernel_h - (in_height % stride[0])))
        pad_w = tvm.select(tvm.all((in_width % stride[1]) == 0), simplify(kernel_w - stride[1]), simplify(kernel_w - (in_width % stride[1])))

        pad_h = tvm.select(tvm.all(pad_h < 0), 0, simplify(pad_h/2))
        pad_w = tvm.select(tvm.all(pad_w < 0), 0, simplify(pad_w/2))

    return (pad_h, pad_w)
