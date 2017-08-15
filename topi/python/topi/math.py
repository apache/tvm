"""Elementwise operators"""
from __future__ import absolute_import as _abs
import tvm

@tvm.tag_scope(tag='ewise')
def identity(x):
    """Take identity of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return tvm.compute(x.shape, lambda *i: x(*i))


@tvm.tag_scope(tag='ewise')
def negation(x):
    """Take negation of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return tvm.compute(x.shape, lambda *i: -x(*i))


@tvm.tag_scope(tag="ewise")
def exp(x):
    """Take exponential of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.exp(x(*i)))


@tvm.tag_scope(tag="ewise")
def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.tanh(x(*i)))


@tvm.tag_scope(tag="ewise")
def log(x):
    """Take logarithm of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.log(x(*i)))


@tvm.tag_scope(tag="ewise")
def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.sqrt(x(*i)))


@tvm.tag_scope(tag="ewise")
def sigmoid(x):
    """Take sigmoid tanh of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.sigmoid(x(*i)))
