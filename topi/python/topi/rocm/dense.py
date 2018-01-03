# pylint: disable=invalid-name, unused-variable
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
import tvm
from tvm.contrib import rocblas
import topi
from ..nn.dense import dense, dense_default
from .. import tag
from .. import generic

@dense.register("rocm")
def dense_rocm(data, weight, bias=None):
    """Dense operator for rocm backend.

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
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    target = tvm.target.current_target()
    if "rocblas" in target.libs:
        matmul = rocblas.matmul(data, weight, False, True)
        if bias is not None:
            matmul = tvm.compute((batch, out_dim), \
                                 lambda i, j: matmul[i, j] + bias[j], \
                                 tag=tag.BROADCAST)
        return matmul
    return dense_default(data, weight, bias)


@generic.schedule_dense.register(["rocm"])
def schedule_dense(outs):
    """Schedule for dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    target = tvm.target.current_target()
    if target.target_name == "rocm" and "rocblas" in target.libs:
        return generic.schedule_extern(outs)
    return topi.cuda.schedule_dense(outs)
