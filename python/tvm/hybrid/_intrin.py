"""Intrinsics of Python-Halide DSL for Python runtime"""

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

serial = unroll = vectorize = parallel = _range #pylint: disable=invalid-name

def allocate(shape, dtype=None):
    """Allocate a buffer with given shape"""
    dtype = 'float32' if dtype is None else dtype
    return numpy.zeros(shape).astype(dtype)

def popcount(x):
    cnt = 0
    while x:
        x -= x & -x
        cnt += 1
    return cnt

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

HYBRID_GLOBALS = {
    'serial'    : serial,
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
    'serial'   : For.Serial,
    'unroll'   : For.Unrolled,
    'parallel' : For.Parallel,
    'vectorize': For.Vectorized,
    'bind'     : None
}

MATH_INTRIN = ['sqrt', 'log', 'exp', 'tanh', 'sigmoid', 'power', 'popcount']
