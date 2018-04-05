# pylint: disable=invalid-name
"""TVM operator for local response norm compute."""
from __future__ import absolute_import
import tvm
import topi
from .pad import pad

@tvm.target.generic_func
def lrn_nchw(data, size, alpha=0.0001, beta=0.75, bias=2):
    """Perform the across channels local response normalisation
    on the input data.

    sum_sqr_up^i{x, y} = (bias+((alpha/size)* \
                                {sum_{j=max(0, i-size/2)}^{min(N-1,i+size/2)} \
                                     (data^j{x,y})^2}))^beta
    output^i{x, y} = data^i{x, y}/sum_sqr_up^i{x, y}
    N is the number for input channels

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
    pad_before = [0, (size//2), 0, 0]
    pad_after = [0, (size//2), 0, 0]
    pad_data = pad(data, pad_before, pad_after, name="pad_data")

    rxk = tvm.reduce_axis((0, size), name='rxk')
    sqr_sum = tvm.compute((b, c, h, w), lambda i, l, j, k: tvm.sum(
        tvm.power(pad_data[i, l + rxk, j, k], 2.0),
        axis=rxk))

    sqr_sum_up = tvm.compute((b, c, h, w), lambda i, j, k, l: tvm.power(
        (bias + (alpha * sqr_sum[i, j, k, l] / size)), beta))

    return topi.broadcast_div(data, sqr_sum_up)
