# pylint: disable=invalid-name, trailing-whitespace
"""Relu operation in python"""

def relu_python(a_np):
    """Relu operator.
    Parameters
    ----------
    a_np : numpy.ndarray
        2-D input data

    Returns
    -------
    output_np : numpy.ndarray
        2-D output with same shape
    """
    return a_np * (a_np > 0)
