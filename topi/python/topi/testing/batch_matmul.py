# pylint: disable=invalid-name
"""Batch matmul in python"""
import numpy as np

def batch_matmul(x, y):
    """batch_matmul operator implemented in numpy.

    Parameters
    ----------
    x : numpy.ndarray
        3-D with shape [batch, M, K]

    y : numpy.ndarray
        3-D with shape [batch, N, K]

    Returns
    -------
    out : numpy.ndarray
        3-D with shape [batch, M, N]
    """
    batch, M, _ = x.shape
    N = y.shape[1]
    out = np.zeros((batch, M, N)).astype(x.dtype)
    for i in range(batch):
        out[i] = np.dot(x[i], y[i].T)
    return out
