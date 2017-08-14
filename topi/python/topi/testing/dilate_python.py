# pylint: disable=invalid-name, line-too-long
"""Dilate operation in python"""
import numpy as np


def dilate_python(input_np, strides):
    """Dilate operation.

    Parameters
    ----------
    input_np : numpy.ndarray
        4-D, can be any layout.

    strides : list/tuple of 4 ints
        Dilation stride on each dimension, 1 means no dilation.

    Returns
    -------
    output_np : numpy.ndarray
        4-D, the same layout as Input.
    """
    A, B, C, D = input_np.shape
    sa, sb, sc, sd = strides
    Ao = (A-1)*sa+1
    Bo = (B-1)*sb+1
    Co = (C-1)*sc+1
    Do = (D-1)*sd+1
    output_np = np.zeros(shape=(Ao, Bo, Co, Do))
    output_np[0::sa, 0::sb, 0::sc, 0::sd] = input_np
    return output_np
