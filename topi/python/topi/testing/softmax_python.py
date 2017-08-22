# pylint: disable=invalid-name, trailing-whitespace
"""Softmax operation in python"""
import numpy as np

def softmax_python(a_np):
    """Softmax operator.
    Parameters
    ----------
    a_np : numpy.ndarray
        2-D input data

    Returns
    -------
    output_np : numpy.ndarray
        2-D output with same shape
    """
    assert len(a_np.shape) == 2, "only support 2-dim softmax"
    max_elem = np.amax(a_np, axis=1)
    max_elem = max_elem.reshape(max_elem.shape[0], 1)
    e = np.exp(a_np-max_elem)
    expsum = np.sum(e, axis=1)
    out_np = e / expsum[:, None]
    return out_np
