# pylint: disable=invalid-name
"""TVM operator for l2 normalize"""
from __future__ import absolute_import
import tvm
from .. import cpp

@tvm.target.generic_func
def l2_normalize(data, eps, axis=None):
    """Perform L2 normalization on the input data

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
    return cpp.nn.l2_normalize(data, eps, axis)
