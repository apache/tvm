"""Local response normalization in python"""
import numpy as np

def lrn_nchw_python(a_np, size, bias, alpha, beta):
    """Local response norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    size : int
        normalisation window size

    bias : float
        offset to avoid dividing by 0. constant value

    alpha : float
        contant valie

    beta : float
        exponent constant value


    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, channel, height, weight = a_np.shape
    radius = size / 2
    sqr_sum = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    sqr_sum_up = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    lrn_out = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    def add_dot_values(i, c, j, k):
        for rxk in range(-radius, radius+1):
            if ((c + rxk) < channel) and ((c + rxk) >= 0):
                sqr_sum[i, c, j, k] = sqr_sum[i, c, j, k] + \
                                       (a_np[i, c + rxk, j, k] * \
                                        a_np[i, c + rxk, j, k])
    for i in range(batch):
        for c in range(channel):
            for j in range(height):
                for k in range(weight):
                    add_dot_values(i, c, j, k)
    for i in range(batch):
        for c in range(channel):
            for j in range(height):
                for k in range(weight):
                    sqr_sum_up[i, c, j, k] = \
                        np.power((bias + (alpha * sqr_sum[i, c, j, k] / \
                                          size)), beta)
    for i in range(batch):
        for c in range(channel):
            for j in range(height):
                for k in range(weight):
                    lrn_out[i, c, j, k] = a_np[i, c, j, k] / \
                                           sqr_sum_up[i, c, j, k]

    return lrn_out
