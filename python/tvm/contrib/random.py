"""External function interface to random library."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api


def randint(low, high, size, dtype='int32'):
    assert dtype == 'int32', 'only support int32 for now'
    return _api.extern(size, [],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.random.randint", low, high, outs[0]),
        dtype=dtype)


_init_api("tvm.contrib.random")
