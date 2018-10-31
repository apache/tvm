# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""gather_nd in python"""
import numpy as np

def gather_nd_python(a_np, indices_np):
    """ Python version of GatherND operator

    Parameters
    ----------
    a_np : numpy.ndarray
        Numpy array

    indices_np : numpy.ndarray
        Numpy array

    Returns
    -------
    b_np : numpy.ndarray
        Numpy array
    """
    a_shape = a_np.shape
    indices_np = indices_np.astype('int32')
    indices_shape = indices_np.shape
    assert len(indices_shape) > 1
    assert indices_shape[0] <= len(a_shape)
    b_shape = list(indices_shape[1:])
    for i in range(indices_shape[0], len(a_shape)):
        b_shape.append(a_shape[i])
    b_np = np.zeros(b_shape)
    for idx in np.ndindex(*indices_shape[1:]):
        a_idx = []
        for i in range(indices_shape[0]):
            indices_pos = tuple([i] + list(idx))
            a_idx.append(indices_np[indices_pos])
        b_np[idx] = a_np[tuple(a_idx)]
    return b_np
