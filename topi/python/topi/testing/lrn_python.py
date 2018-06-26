# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""LRN in python"""
from itertools import product
import numpy as np

def lrn_python(a_np, size, axis, bias, alpha, beta):
    """Local response normalization operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    size : int
        normalization window size

    axis : int
        input data layout channel axis

    bias : float
        offset to avoid dividing by 0. constant value

    alpha : float
        constant value

    beta : float
        exponent constant value

    Returns
    -------
    lrn_out : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    radius = size // 2
    sqr_sum = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    for i, j, k, l in product(*[range(_axis) for _axis in a_np.shape]):
        axis_size = a_np.shape[axis]
        if axis == 1:
            #NCHW layout
            sum_start = j-radius if j-radius >= 0 else 0
            sum_end = j+radius+1 if j+radius+1 < axis_size else axis_size
            sqr_sum[i, j, k, l] = sum(a_np[i, sum_start:sum_end, k, l] * \
                                      a_np[i, sum_start:sum_end, k, l])
        elif axis == 3:
            #NHWC layout
            sum_start = l-radius if l-radius >= 0 else 0
            sum_end = l+radius+1 if l+radius+1 < axis_size else axis_size
            sqr_sum[i, j, k, l] = sum(a_np[i, j, k, sum_start:sum_end] * \
                                      a_np[i, j, k, sum_start:sum_end])

    sqr_sum_up = np.power((bias + (alpha * sqr_sum /size)), beta)
    lrn_out = np.divide(a_np, sqr_sum_up)
    return lrn_out
