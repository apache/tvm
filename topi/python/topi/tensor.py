# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""
from __future__ import absolute_import as _abs
import tvm
from . import tag

@tvm.tag_scope(tag=tag.ELEMWISE)
def elemwise_sum(xs, num_args):
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.Tensor
        Input arguments.
    num_args : int
        Number of arguments

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    assert len(xs) > 0, "elemwise sum must have at least one input tensor."

    def _compute(*i):
        return sum([x(*i) for x in xs])

    return tvm.compute(xs[0].shape, _compute)


@tvm.tag_scope(tag=tag.ELEMWISE)
def full(shape, dtype, fill_value):
    """Fill tensor with fill_value

    Parameters
    ----------
    shape : tuple
        Input tensor shape.
    dtype : str
        Data type
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(shape, lambda *i: tvm.const(fill_value, dtype))


@tvm.tag_scope(tag=tag.ELEMWISE)
def full_like(x, fill_value):
    """Construct a tensor with same shape as input tensor,
       then fill tensor with fill_value.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    dtype = x.dtype
    return tvm.compute(x.shape, lambda *i: tvm.const(fill_value, dtype))
