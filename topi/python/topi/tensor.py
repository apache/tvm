# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""
from __future__ import absolute_import as _abs
import tvm
from . import cpp
from . import tag

@tvm.tag_scope(tag=tag.ELEMWISE)
def elemwise_sum(xs):
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.Tensor
        Input arguments.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return cpp.elemwise_sum(xs)


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
    return cpp.full(shape, dtype, fill_value)


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
    return cpp.full_like(x, fill_value)
