"""Binary Neural Network (BNN) Operators"""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import tvm
from ..util import get_const_tuple


def batch_matmul(x, y):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch.

    Parameters
    ----------
    x : tvm.Tensor
        3-D with shape [batch, M, K]

    y : tvm.TEnsor
        3-D with shape [batch, N, K]

    Returns
    -------
    output : tvm.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_matmul"
    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    assert x_shape[0] == y_shape[0], "batch dimension doesn't match"
    assert x_shape[2] == y_shape[2], "shapes of x and y is inconsistant"
    batch, M, K = x.shape
    N = y.shape[1]
    k = tvm.reduce_axis((0, K), name='k')
    return tvm.compute((batch, M, N),
                       lambda b, i, j: tvm.sum(x[b, i, k] * y[b, j, k], axis=k),
                       tag='batch_matmul')
