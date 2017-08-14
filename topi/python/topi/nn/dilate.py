# pylint: disable=invalid-name, line-too-long
"""Dilation operators"""
from __future__ import absolute_import as _abs
import tvm


@tvm.tag_scope(tag="dilation")
def dilate(Input, strides):
    """Dilate Input with zeros.

    Parameters
    ----------
    Input : tvm.Tensor
        n-D, can be any layout.

    strides : list / tuple of n ints
        Dilation stride on each dimension, 1 means no dilation.

    Returns
    -------
    Output : tvm.Tensor
        n-D, the same layout as Input.
    """
    n = len(Input.shape)
    assert n <= 5, \
        "Dimension of input tensor cannot exceed 5"
    assert len(strides) == n, \
        "Input dimension and strides size dismatch : %d vs %d" %(n, len(strides))
    output_size = ()
    for i in range(n):
        output_size += (tvm.ir_pass.Simplify((Input.shape[i]-1)*strides[i]+1),)

    if n == 5:
        Output = tvm.compute(
            (output_size),
            lambda *i: tvm.select(
                tvm.all((i[0]%strides[0]).equal(0),
                        (i[1]%strides[1]).equal(0),
                        (i[2]%strides[2]).equal(0),
                        (i[3]%strides[3]).equal(0),
                        (i[4]%strides[4]).equal(0)),
                Input(i[0]/strides[0], i[1]/strides[1], i[2]/strides[2], i[3]/strides[3], i[4]/strides[4]),
                tvm.const(0.0, Input.dtype)), name='DilatedInput')
    elif n == 4:
        Output = tvm.compute(
            (output_size),
            lambda *i: tvm.select(
                tvm.all((i[0]%strides[0]).equal(0),
                        (i[1]%strides[1]).equal(0),
                        (i[2]%strides[2]).equal(0),
                        (i[3]%strides[3]).equal(0)),
                Input(i[0]/strides[0], i[1]/strides[1], i[2]/strides[2], i[3]/strides[3]),
                tvm.const(0.0, Input.dtype)), name='DilatedInput')
    elif n == 3:
        Output = tvm.compute(
            (output_size),
            lambda *i: tvm.select(
                tvm.all((i[0]%strides[0]).equal(0),
                        (i[1]%strides[1]).equal(0),
                        (i[2]%strides[2]).equal(0)),
                Input(i[0]/strides[0], i[1]/strides[1], i[2]/strides[2]),
                tvm.const(0.0, Input.dtype)), name='DilatedInput')
    elif n == 2:
        Output = tvm.compute(
            (output_size),
            lambda *i: tvm.select(
                tvm.all((i[0]%strides[0]).equal(0),
                        (i[1]%strides[1]).equal(0)),
                Input(i[0]/strides[0], i[1]/strides[1]),
                tvm.const(0.0, Input.dtype)), name='DilatedInput')
    else: # n == 1
        Output = tvm.compute(
            (output_size),
            lambda *i: tvm.select(
                tvm.all((i[0]%strides[0]).equal(0)),
                Input(i[0]/strides[0]),
                tvm.const(0.0, Input.dtype)), name='DilatedInput')

    return Output
