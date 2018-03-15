"""Elementwise operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from ..broadcast import broadcast_mul

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

def prelu(x, weights):
    """ PReLU.
    It accepts two arguments: an input ``x`` and a weight array ``W``
    and computes the output as :math:`PReLU(x) = \\max(x, W*x)`,
    where :math:`*` is an elementwise multiplication for each sample in the
    batch.
    Arguments:
    x : tvm.Tensor
        Input argument.

    w : tvm.Tensor
        Weight Tensor.

    Returns:
    y : tvm.Tensor
        The result.

    Links:
        [http://arxiv.org/pdf/1502.01852v1.pdf]
    """

    m = broadcast_mul(x, weights)
    def _compute(*indices):
        return tvm.select(x(*indices) > 0, x(*indices), m(*indices))
    return tvm.compute(x.shape, _compute)
