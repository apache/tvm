"""Intrinsics of Python-Halide DSL for Python runtime"""

import numpy

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

class bind(_range):
    def __init__(self, ext, tag):
        super(range, self).__init__(ext)
        self.tag = tag

serial = unrolled = vectorized = parallel = _range #pylint: disable=invalid-name

def allocate(shape, dtype=None):
    return numpy.zeros(shape).astype(dtype)

HYBRID_GLOBALS = {
    'serial'    : serial,
    'unrolled'  : unrolled,
    'vectorized': vectorized,
    'parallel'  : parallel,
    'allocate'  : allocate,
    'bind'      : bind
}

LOOP_INTRIN = ['serial', 'unrolled', 'parallel', 'vectorized', 'bind']
