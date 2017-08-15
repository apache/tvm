# pylint: disable=invalid-name
"""Dilate operation in python"""
import numpy as np


def dilate_python(input_np, strides):
    """Dilate operation.

    Parameters
    ----------
    input_np : numpy.ndarray
        n-D, can be any layout.

    strides : list / tuple of n ints
        Dilation stride on each dimension, 1 means no dilation.

    Returns
    -------
    output_np : numpy.ndarray
        n-D, the same layout as Input.
    """
    n = len(input_np.shape)
    assert len(strides) == n, \
        "Input dimension and strides size dismatch : %d vs %d" %(n, len(strides))
    output_size = ()
    no_zero = ()
    for i in range(n):
        output_size += ((input_np.shape[i]-1)*strides[i]+1,)
        no_zero += ((range(0, output_size[i], strides[i])),)
    output_np = np.zeros(shape=output_size)
    output_np[np.ix_(*no_zero)] = input_np

    return output_np
