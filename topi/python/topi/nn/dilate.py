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
        n-D, can be any layout.

    strides : list / tuple of n ints
        Dilation stride on each dimension, 1 means no dilation.

    Returns
    -------
    Output : tvm.Tensor
        n-D, the same layout as Input.
    """
    n = len(Input.shape)
    assert len(strides) == n, \
        "Input dimension and strides size dismatch : %d vs %d" %(n, len(strides))
    output_size = ()
    for i in range(n):
        output_size += (tvm.ir_pass.Simplify((Input.shape[i]-1)*strides[i]+1),)

    def _dilate(data, *indices):
        not_zero = (indices[0]%strides[0]).equal(0)
        index_tuple = ()
        for i in range(n):
            index_tuple += (indices[i]/strides[i],)
            not_zero = tvm.all(not_zero, (indices[i]%strides[i]).equal(0))
        return tvm.select(not_zero, data[index_tuple], tvm.const(0.0, data.dtype))

    Output = tvm.compute(
        (output_size),
        lambda *indices: _dilate(Input, *indices),
        name='DilatedInput')

    return Output
