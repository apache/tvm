# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""L2 normalize in python"""
import numpy as np

def l2_normalize_python(a_np, eps, axis=None):
    """L2 normalize operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    eps : float
        epsilon constant value
    axis : list of int
        axis over the normalization applied

    Returns
    -------
    l2_normalize_out : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    dot_value = np.power(a_np, 2.0)
    sqr_sum = np.sum(dot_value, axis, keepdims=True)
    sqrt_sum = np.sqrt(np.maximum(np.broadcast_to(sqr_sum, a_np.shape), eps))
    l2_normalize_out = np.divide(a_np, sqrt_sum)
    return l2_normalize_out
