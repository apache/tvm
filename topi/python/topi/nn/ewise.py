"""Elementwise operators"""
from __future__ import absolute_import as _abs
import tvm

@tvm.tag_scope(tag="ewise")
def relu(x):
    """Take relu of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.max(x(*i), 0))
