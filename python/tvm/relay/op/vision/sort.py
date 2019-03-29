"""Argsort operation"""
from __future__ import absolute_import as _abs
from . import _make

def argsort(data, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array of indicies
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    valid_count : tvm.Tensor
        The number of valid elements to be sorted.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : relay.Expr
        Tensor with same shape as data.
    """
    return _make.argsort(data, axis, is_ascend, dtype)
