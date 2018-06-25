"""Intrinsics of TVM-Python Hybrid Script for Python runtime"""

import numpy
from ..stmt import For

class _range(object):
    """Base class of the loop ranges in hybrid script"""
    def __init__(self, a, b=None):
        if b is None:
            self.low = 0
            self.ext = a
        else:
            self.low = a
            self.ext = b

    def __iter__(self):
        i = 0
        while i < self.ext:
            yield i + self.low
            i += 1


class bind(_range): #pylint: disable=invalid-name
    def __init__(self, tag, ext):
        super(bind, self).__init__(ext)
        self.tag = tag


unroll = vectorize = parallel = _range #pylint: disable=invalid-name


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
    'unroll'    : unroll,
    'vectorize' : vectorize,
    'parallel'  : parallel,
    'allocate'  : allocate,
    'bind'      : bind,
    'sqrt'      : numpy.sqrt,
    'log'       : numpy.log,
    'tanh'      : numpy.tanh,
    'power'     : numpy.power,
    'exp'       : numpy.exp,
    'sigmoid'   : sigmoid,
    'popcount'  : popcount
}


LOOP_INTRIN = {
    'range'    : For.Serial,
    'unroll'   : For.Unrolled,
    'parallel' : For.Parallel,
    'vectorize': For.Vectorized,
    'bind'     : None
}


MATH_INTRIN = ['sqrt', 'log', 'exp', 'tanh', 'sigmoid', 'power', 'popcount']
