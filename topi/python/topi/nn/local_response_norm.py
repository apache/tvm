# pylint: disable=invalid-name
"""TVM operator for local response norm compute."""
from __future__ import absolute_import
import tvm
from .pad import pad

@tvm.target.generic_func
def lrn_nchw(data, size, alpha=0.0001, beta=0.75, bias=2):
    """Perform local response normalisation on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    size : int
        normalisation window size

    bias : float
        offset to avoid dividing by 0

    alpha : float
        to be divided

    beta : float
        exponent

    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    assert len(data.shape) == 4, "only support 4-dim lrn"
    b, c, h, w = data.shape
    assert (size % 2) == 1, "size should be odd number"

    ##Add padding on left & right of size radius first
    pad_before = [0, int(size/2), 0, 0]
    pad_after = [0, int(size/2), 0, 0]
    pad_data = pad(data, pad_before, pad_after, name="pad_data")

    rxk = tvm.reduce_axis((0, size), name='rxk')
    sqr_sum = tvm.compute((b, c, h, w), lambda i, ll, j, k: tvm.sum(
        pad_data[i, ll + rxk, j, k] * pad_data[i, ll + rxk, j, k],
        axis=rxk))

    sqr_sum_up = tvm.compute((b, c, h, w), lambda i, j, k, l: tvm.intrin.power(
        (bias + (alpha * sqr_sum[i, j, k, l] / size)), beta))

    return tvm.compute(data.shape,
                       lambda b, c, h, w: data[b, c, h, w] / sqr_sum_up[b, c, h, w])
