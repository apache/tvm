"""Elementwise operators"""
from __future__ import absolute_import as _abs
import tvm
from . import tag
from .util import get_const_tuple

@tvm.tag_scope(tag=tag.ELEMWISE)
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


@tvm.tag_scope(tag=tag.ELEMWISE)
def negative(x):
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


@tvm.tag_scope(tag=tag.ELEMWISE)
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


@tvm.tag_scope(tag=tag.ELEMWISE)
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


@tvm.tag_scope(tag=tag.ELEMWISE)
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


@tvm.tag_scope(tag=tag.ELEMWISE)
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


@tvm.tag_scope(tag=tag.ELEMWISE)
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


@tvm.tag_scope(tag=tag.ELEMWISE)
def left_shift(x, n):
    """Take n bits left shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: x(*i) << n)


@tvm.tag_scope(tag=tag.ELEMWISE)
def right_shift(x, n):
    """Take n bits right shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: x(*i) >> n)


@tvm.tag_scope(tag=tag.ELEMWISE)
def clip(x, a_min, a_max):
    """Clip (limit) the values in an array. Given an interval, values
    outside the interval are clipped to the interval edges.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    a_min : int or float
        Minimum value.
    a_max : int or float
        Maximum value.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    def _compute(*indices):
        value = x(*indices)
        const_min = tvm.const(a_min, value.dtype)
        const_max = tvm.const(a_max, value.dtype)
        return tvm.max(tvm.min(value, const_max), const_min)
    return tvm.compute(x.shape, _compute)


@tvm.tag_scope(tag=tag.ELEMWISE)
def cast(x, dtype):
    """Cast input to specified data type.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    dtype : str
        Data type.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: x(*i).astype(dtype))


@tvm.target.generic_func
def matmul(a, b):
    """Multiply matrix a by matrix b, producing a * b

       matmul(x,y) = sum(x[i,j,:]*y[:,a,b])

    Parameters
    ----------
    a : tvm.Tensor
        Left matrix.
    b : tvm.Tensor
        Right matrix.

    Returns
    -------
    ret : tvm.Tensor
        The result matrix.
    """
    if len(a.shape) == 1:
        a = tvm.compute((1,) + get_const_tuple(a.shape), lambda x, y: a[y])

    l_shape = get_const_tuple(a.shape)

    if len(b.shape) == 1:
        b = tvm.compute((1,) + get_const_tuple(b.shape), lambda x, y: b[y])

    r_shape = get_const_tuple(b.shape)

    l_dim = len(l_shape)
    out_shape = l_shape[:-1] + r_shape[1:]

    k = tvm.reduce_axis((0, l_shape[-1]), name='k')
    assert l_shape[-1] == r_shape[0], "shape inconsistent %d vs %d" % \
                                      (l_shape[-1], r_shape[0])
    out = tvm.compute(out_shape,
                      lambda *idx: tvm.sum(
                          a[idx[:l_dim - 1] + (k,)] *
                          b[(k,) + idx[l_dim - 1:]],
                          axis=k
                      ),
                      tag="matmul")
    return out
