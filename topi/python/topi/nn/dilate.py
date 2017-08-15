# pylint: disable=invalid-name
"""Dilation operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import util


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
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not util.equal_const_int(strides[i], 1):
                index_tuple.append(indices[i]/strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.select(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    Output = tvm.compute(
        output_size,
        lambda *indices: _dilate(Input, *indices),
        name='DilatedInput')

    return Output
