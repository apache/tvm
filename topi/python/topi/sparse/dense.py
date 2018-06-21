"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag

def dense_default(data, indices, indptr, weight, bias=None):
    # pylint: disable=unused-argument
    """The default implementation of dense in topi.

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
    assert len(data.shape) == 1 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    # assert data.stype == 'csr', \
    #     "data matrix is assumed to be sparse matrix, but data is `%s`" % (type(data),)
    assert isinstance(weight, tvm.tensor.Tensor), \
        "weight matrix is assumed to be tvm.Tensor, but weight is `%s`" % (type(weight))
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = 1, data.shape[0]
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    # matmul = tvm.compute((batch, out_dim), \
    #                      lambda i, j: tvm.sum(data.data[i, k] * weight[j, k], axis=k), \
    #                      tag='dense')
    matmul = tvm.compute((batch, out_dim), \
                         lambda i, j: tvm.sum(data[i] * weight[i, k], axis=k), \
                         tag='spmm')
    print(matmul.op.body)
    if bias is not None:
        matmul = tvm.compute((batch, out_dim), \
                             lambda i, j: matmul[i, j] + bias[j], \
                             tag=tag.BROADCAST)
    return matmul


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
    return dense_default(data.data, data.indices, data.indptr, weight, bias)
