"""Reshaping tensor operations."""

from . import _make


def expand_dims(data, axis, num_newaxis=1):
    """TODO(Junru): docstring
    """
    return _make.expand_dims(data, axis, num_newaxis)
