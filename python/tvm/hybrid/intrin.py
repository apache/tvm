"""Intrinsics of TVM-Python Hybrid Script for Python emulation runtime"""

import numpy


class bind(object): #pylint: disable=invalid-name
    """GPU bind software emulataion runtime."""
    def __init__(self, _, ext):
        self.ext = ext

    def __iter__(self):
        i = 0
        while i < self.ext:
            yield i
            i += 1


def allocate(shape, dtype='float32', scope='global'): #pylint: disable=unused-argument
    """Allocate a buffer with given shape

    Parameters
    ----------
    shape: Tuple
        The shape of the tensor to be allocated
    dtype: string
        The data type of the tensor
    scope: string
        The storage scope of the tensor

    Returns
    -------
    tensor: numpy.array
        The tensor allocated
    """
    return numpy.zeros(shape).astype(dtype)


def popcount(x):
    """
    Count ones in the binary representation of number x

    Parameters
    ----------
    x: Integer
        The number to be counted

    Returns
    -------
    cnt: Integer
        The number of ones in the binary representation of number x
    """
    cnt = 0
    while x:
        x -= x & -x
        cnt += 1
    return cnt


def sigmoid(x):
    """
    Sigmoid function of x, aka 1/(1+exp(-x)).

    Parameters
    ----------
    x: a real number

    Returns
    -------
    res: a real number
        The result of sigmoid function
    """
    return 1 / (1 + numpy.exp(-x))


HYBRID_GLOBALS = {
    'len'          : len,
    'unroll'       : range,
    'vectorize'    : range,
    'parallel'     : range,
    'const_range'  : range,
    'bind'         : bind,
    'allocate'     : allocate,
    'output_tensor': allocate,
    'sqrt'         : numpy.sqrt,
    'log'          : numpy.log,
    'tanh'         : numpy.tanh,
    'power'        : numpy.power,
    'exp'          : numpy.exp,
    'sigmoid'      : sigmoid,
    'popcount'     : popcount,
}
