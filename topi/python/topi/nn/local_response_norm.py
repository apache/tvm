# pylint: disable=invalid-name
"""TVM operator for local response norm compute."""
from __future__ import absolute_import
import tvm
import topi
from .pad import pad

@tvm.target.generic_func
def lrn(data, size, axis=1, alpha=0.0001, beta=0.75, bias=2):
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

    axis : int
        input data layout channel axis
        default value is 1 for NCHW format

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
    assert (size % 2) == 1, "size should be odd number"
    assert (axis == 1) or (axis == 3), "axis should 1 or 3 for NCHW and NHWC"
    ##Add padding on left & right of size radius first
    pad_after = pad_before = [0, 0, 0, 0]
    pad_after[axis] = pad_before[axis] = (size//2)
    pad_data = pad(data, pad_before, pad_after, name="pad_data")

    rxs = tvm.reduce_axis((0, size), name='rxs')
    if axis == 1:
        #NCHW layout
        sqr_sum = tvm.compute(data.shape, lambda i, j, k, l: tvm.sum(
            pad_data[i, j + rxs, k, l] * pad_data[i, j + rxs, k, l],
            axis=rxs))
    elif axis == 3:
        #NHWC layout
        sqr_sum = tvm.compute(data.shape, lambda i, j, k, l: tvm.sum(
            pad_data[i, j, k, l + rxs] * pad_data[i, j, k, l + rxs],
            axis=rxs))

    sqr_sum_up = tvm.compute(data.shape, lambda i, j, k, l: tvm.power(
        (bias + (alpha * sqr_sum[i, j, k, l] / size)), beta))

    return topi.broadcast_div(data, sqr_sum_up)
