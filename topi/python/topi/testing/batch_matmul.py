# pylint: disable=invalid-name
"""Batch matmul in python"""
import numpy as np

def batch_matmul(x, y):
    batch, M, K = x.shape
    N = y.shape[1]
    out = np.zeros((batch, M, N)).astype(x.dtype)
    for i in range(batch):
        out[i] = np.dot(x[i], y[i].T)
    return out
