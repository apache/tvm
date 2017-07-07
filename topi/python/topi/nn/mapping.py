# pylint: disable=invalid-name, line-too-long
"""Operators of one-to-one-mapping on the first input"""
from __future__ import absolute_import as _abs
import tvm

@tvm.tag_scope(tag="scale_shift")
def scale_shift(Input, Scale, Shift):
    """Batch normalization operator in inference.

    Parameters
    ----------
    Input : tvm.Tensor
        Input tensor, layout is NCHW

    Scale : tvm.Tensor
        Scale tensor, 1-D of size channel number

    Shift : tvm.Tensor
        Shift tensor, 1-D of size channel number

    Returns
    -------
    Output : tvm.Tensor
        Output tensor, layout is NCHW
    """
    return tvm.compute(Input.shape, lambda b, c, i, j: Input[b, c, i, j] * Scale[c] + Shift[c], name='ScaleShift')
