"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm


@tvm.tag_scope(tag='fully_connected')
def fully_connected(data, weight):
    """Matrix multiplication

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim fully_connected"
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    return tvm.compute((batch, out_dim), lambda i, j: \
        tvm.sum(data[i][k] * weight[j][k], axis=k))


@tvm.tag_scope(tag='fully_connected_with_bias')
def fully_connected_with_bias(data, weight, bias):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim fully_connected"
    assert len(data.shape) == 2 and len(weight.shape) == 2 and len(bias.shape) == 1, \
        "only support 2-dim fully_connected"
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    matmul = tvm.compute((batch, out_dim), lambda i, j: \
        tvm.sum(data[i, k] * weight[j, k], axis=k))
    return tvm.compute((batch, out_dim), lambda i, j: \
        matmul[i, j] + bias[j])
