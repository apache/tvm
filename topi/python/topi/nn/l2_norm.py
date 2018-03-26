# pylint: disable=invalid-name
"""TVM operator for l2norm"""
from __future__ import absolute_import
import tvm

@tvm.target.generic_func
def l2norm_instance(data, eps):
    """Perform L2norm on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with NCHW or NHWC layout

    eps : float
        epsilon value


    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    assert len(data.shape) == 4, "only support 4-dim lrn"
    b, x1, x2, x3 = data.shape

    rx1 = tvm.reduce_axis((0, x1), name='rx1')
    rx2 = tvm.reduce_axis((0, x2), name='rx2')
    rx3 = tvm.reduce_axis((0, x3), name='rx3')
    sqr_sum = tvm.compute((b),
                          lambda i: tvm.sum(data[i, rx1, rx2, rx3] * \
                                                data[i, rx1, rx2, rx3],
                                            axis=(rx1, rx2, rx3)))
    sqrt_sum = tvm.compute((b), lambda i: tvm.sqrt(sqr_sum[i] + eps))
    return tvm.compute(
        data.shape, lambda b, c, h, w: data[b, c, h, w] / sqrt_sum[b])
