# pylint: disable=invalid-name, line-too-long
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
    assert n <= 5, \
        "Dimension of input tensor cannot exceed 5"
    assert len(strides) == n, \
        "Input dimension and strides size dismatch : %d vs %d" %(n, len(strides))
    output_size = ()
    for i in range(n):
        output_size += ((input_np.shape[i]-1)*strides[i]+1,)
    output_np = np.zeros(shape=output_size)

    if n == 5:
        output_np[0::strides[0], 0::strides[1], 0::strides[2], 0::strides[3], 0::strides[4]] = input_np
    elif n == 4:
        output_np[0::strides[0], 0::strides[1], 0::strides[2], 0::strides[3]] = input_np
    elif n == 3:
        output_np[0::strides[0], 0::strides[1], 0::strides[2]] = input_np
    elif n == 2:
        output_np[0::strides[0], 0::strides[1]] = input_np
    else: # n == 1
        output_np[0::strides[0]] = input_np

    return output_np
