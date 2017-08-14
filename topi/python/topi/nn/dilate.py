# pylint: disable=invalid-name
"""Dilation operators"""
from __future__ import absolute_import as _abs
import tvm


@tvm.tag_scope(tag="dilation")
def dilate(Input, strides):
    """Dilate Input with zeros.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D, can be any layout.

    strides : list/tuple of 4 ints
        Dilation stride on each dimension, 1 means no dilation.

    Returns
    -------
    Output : tvm.Tensor
        4-D, the same layout as Input.
    """
    A, B, C, D = Input.shape
    sa, sb, sc, sd = strides
    Output = tvm.compute(
        ((A-1)*sa+1, (B-1)*sb+1, (C-1)*sc+1, (D-1)*sd+1),
        lambda a, b, c, d: tvm.select(
            tvm.all((a%sa).equal(0), (b%sb).equal(0), (c%sc).equal(0), (d%sd).equal(0)),
            Input(a/sa, b/sb, c/sc, d/sd), tvm.const(0.0, Input.dtype)),
        name='DilatedInput')
    return Output
