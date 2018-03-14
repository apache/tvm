# pylint: disable=invalid-name
"""TVM operator for l2norm"""
from __future__ import absolute_import
import tvm

@tvm.target.generic_func
def l2norm_instance_nchw(data, eps):
    """Perform local response normalisation on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    eps : float
        epsilon value


    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    assert len(data.shape) == 4, "only support 4-dim lrn"
    b, c, h, w = data.shape

    rxh = tvm.reduce_axis((0, h), name='rxh')
    rxw = tvm.reduce_axis((0, w), name='rxw')
    rxc = tvm.reduce_axis((0, c), name='rxc')
    sqr_sum = tvm.compute((b),
                          lambda i: tvm.sum(data[i, rxc, rxh, rxw] * \
                                                data[i, rxc, rxh, rxw],
                                            axis=(rxc, rxh, rxw)))
    sqrt_sum = tvm.compute((b), lambda i: tvm.sqrt(sqr_sum[i] + eps))
    return tvm.compute(
        data.shape, lambda b, c, h, w: data[b, c, h, w] / sqrt_sum[b])
