"""External function interface to random library."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api


def randint(low, high, size, dtype='int32'):
    return _api.extern(size, [],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.random.randint", low, high, outs[0]),
        dtype=dtype)


def uniform(low, high, size):
    return _api.extern(size, [],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.random.uniform", float(low), float(high), outs[0]),
        dtype='float32')

_init_api("tvm.contrib.random")
