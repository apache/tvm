"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag


def dense(data, weight, bias=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    matmul = tvm.compute((batch, out_dim), \
                         lambda i, j: tvm.sum(data[i, k] * weight[j, k], axis=k), \
                         tag='dense')
    if bias:
        matmul = tvm.compute((batch, out_dim), \
                             lambda i, j: matmul[i, j] + bias[j], \
                             tag=tag.BROADCAST)
    return matmul
