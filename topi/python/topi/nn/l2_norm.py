# pylint: disable=invalid-name
"""TVM operator for l2norm"""
from __future__ import absolute_import
import tvm
import topi

@tvm.target.generic_func
def l2norm_instance(data, eps, axis=None):
    """Perform L2norm on the input data

    For axis=None, y(i, j) = x(i, j) / sqrt(max(sum(x^2), eps))

    Parameters
    ----------
    data : tvm.Tensor
        4-D with NCHW or NHWC layout

    eps : float
        epsilon value

    axis : list of int
        axis over the normalization applied

    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    assert len(data.shape) == 4, "only support 4-dim lrn"
    dot_value = topi.cpp.pow(data, 2.0)
    sum_value = topi.sum(dot_value, axis=axis, keepdims=True)
    expand_sum = topi.broadcast_to(sum_value, data.shape)
    return topi.broadcast_div(data, topi.sqrt(\
                tvm.compute(expand_sum.shape, lambda i, j, k, l:\
                tvm.max(expand_sum[i, j, k, l], eps), tag='l2norm')))
