"""Elementwise operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from ..util import get_const_int

@tvm.tag_scope(tag=tag.ELEMWISE)
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
    return tvm.compute(x.shape, lambda *i: tvm.max(x(*i), tvm.const(0, x.dtype)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def leaky_relu(x, alpha):
    """Take leaky relu of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    alpha : float
        The slope for the small gradient when x < 0

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    def _compute(*indices):
        value = x(*indices)
        calpha = tvm.const(alpha, value.dtype)
        return tvm.select(value > 0, value, value * calpha)
    return tvm.compute(x.shape, _compute)

@tvm.tag_scope(tag=tag.BROADCAST)
def prelu(x, slope, axis=1):
    """ PReLU.
    It accepts two arguments: an input ``x`` and a weight array ``W``
    and computes the output as :math:`PReLU(x) y = x > 0 ? x : W * x`,
    where :math:`*` is an elementwise multiplication for each sample in the
    batch.
    Arguments:
    x : tvm.Tensor
        Input argument.

    slope : tvm.Tensor
        Channelised slope tensor for prelu

    axis : int
        The axis where the channel data needs to be applied

    Returns:
    y : tvm.Tensor
        The result.

    Links:
        [http://arxiv.org/pdf/1502.01852v1.pdf]
    """

    assert len(x.shape) == 4 and len(slope.shape) == 1
    assert axis < len(x.shape)
    assert get_const_int(slope.shape[0]) == get_const_int(x.shape[axis])

    def _compute_channelwise(*indices):
        return tvm.select(x(*indices) > 0, x(*indices), x(*indices) * slope(indices[axis]))
    return tvm.compute(x.shape, _compute_channelwise)
