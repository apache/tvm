"""L2 norm in python"""
import numpy as np

def l2norm_nchw_python(a_np, eps):
    """L2 norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    eps : float
        epsilon constant value

    Returns
    -------
    l2norm_out : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, channel, height, weight = a_np.shape
    sqr_sum = np.zeros(shape=(batch,)).astype(a_np.dtype)
    sqrt_sum = np.zeros(shape=(batch,)).astype(a_np.dtype)
    l2norm_out = np.zeros(shape=a_np.shape).astype(a_np.dtype)

    for i in range(batch):
        for j in range(channel):
            for k in range(height):
                for m in range(weight):
                    sqr_sum[i] = sqr_sum[i] + (a_np[i, j, k, m] * \
                                               a_np[i, j, k, m])
    for b in range(batch):
        sqrt_sum[b] = np.sqrt(sqr_sum[b] + eps)
    for b in range(batch):
        for j in range(channel):
            for k in range(height):
                for m in range(weight):
                    l2norm_out[b, j, k, m] = a_np[b, j, k, m]/sqrt_sum[b]
    return l2norm_out
