"""Intrinsics of TVM-Python Hybrid Script for Python emulation runtime"""

import numpy
from .. import target


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


def max_num_threads(allow_none=True):
    """Get max number of threads for GPU targets."""
    return target.current_target(allow_none).max_num_threads


HYBRID_GLOBALS = {
    'unroll'         : range,
    'vectorize'      : range,
    'parallel'       : range,
    'const_range'    : range,
    'bind'           : bind,
    'allocate'       : allocate,
    'output_tensor'  : allocate,
    'sqrt'           : numpy.sqrt,
    'log'            : numpy.log,
    'tanh'           : numpy.tanh,
    'power'          : numpy.power,
    'exp'            : numpy.exp,
    'sigmoid'        : sigmoid,
    'popcount'       : popcount,
    'likely'         : lambda cond: cond,
    'uint8'          : numpy.uint8,
    'uint16'         : numpy.uint16,
    'uint32'         : numpy.uint32,
    'uint64'         : numpy.uint64,
    'int8'           : numpy.int8,
    'int16'          : numpy.int16,
    'int32'          : numpy.int32,
    'int64'          : numpy.int64,
    'float16'        : numpy.float16,
    'float32'        : numpy.float32,
    'float64'        : numpy.float64,
    'ceil_div'       : lambda a, b: (a + b - 1) // b,
    'max_num_threads': max_num_threads
}


def _enter_hybrid_runtime(func):
    """Put hybrid runtime variables into the global scope"""
    _globals = func.__globals__
    intersect = []
    for elem in list(HYBRID_GLOBALS.keys()):
        if elem in _globals.keys():
            intersect.append((elem, _globals[elem]))
        _globals[elem] = HYBRID_GLOBALS[elem]
    return intersect


def _restore_runtime(func, intersect):
    """Rollback the modification caused by hybrid runtime"""
    _globals = func.__globals__
    for elem in list(HYBRID_GLOBALS.keys()):
        _globals.pop(elem)
    for k, v in intersect:
        _globals[k] = v
