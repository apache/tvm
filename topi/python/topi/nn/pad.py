"""Pad the data by constant value """
from __future__ import absolute_import as _abs
import tvm
from ..util import equal_const_int


def _spatial2d_pad_option(padding, kernel):
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


@tvm.tag_scope(tag="pad")
def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
    """Dilate Input with zeros.

    Parameters
    ----------
    data : tvm.Tensor
        n-D input, can be any layout.

    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.

    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.

    pad_value : float, optional
        The value to be padded.

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : tvm.Tensor
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError("Input dimension and pad_before dismatch : %d vs %d" % (
            n, len(pad_before)))
    if len(pad_after) != n:
        raise ValueError("Input dimension and pad_after dismatch : %d vs %d" % (
            n, len(pad_before)))
    out_shape = tuple(
        tvm.ir_pass.Simplify(
            (data.shape[i] + pad_before[i] + pad_after[i])) for i in range(n))
    pad_value = (pad_value if isinstance(pad_value, tvm.expr.Expr)
                 else tvm.const(pad_value, data.dtype))
    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.select(not_zero, data(*index_tuple), pad_value)
        return data(*index_tuple)
    return tvm.compute(out_shape, _pad, name=name)
