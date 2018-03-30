"""External function interface to random library."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api


def randint(low, high, size, dtype='int32'):
    """Return random integers from low (inclusive) to high (exclusive).
    Return random integers from the "discrete uniform" distribution of the
    specified dtype in the "half-open" interval [low, high).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution
    high : int
        One above the largest (signed) integer to be drawn from the distribution

    Returns
    -------
    out : Tensor
        A tensor with specified size and dtype
    """
    assert 'int' in dtype, "the type of randint output must be int or uint"
    return _api.extern(size, [], lambda ins, outs: _intrin.call_packed(
        "tvm.contrib.random.randint", int(low), int(high), outs[0]), dtype=dtype)


def uniform(low, high, size):
    """Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval [low, high)
    (includes low, but excludes high). In other words, any value within the
    given interval is equally likely to be drawn by uniform.

    Parameters
    ----------
    low : float
        Lower boundary of the output interval. All values generated will be
        greater than or equal to low.
    high : float
        Upper boundary of the output interval. All values generated will be
        less than high.
    size : tuple of ints
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn.

    Returns
    -------
    out : Tensor
        A tensor with specified size and dtype.
    """
    return _api.extern(size, [], lambda ins, outs: _intrin.call_packed(
        "tvm.contrib.random.uniform", float(low), float(high), outs[0]), dtype='float32')

_init_api("tvm.contrib.random")
