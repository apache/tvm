# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface of expr function exposed from C++."""
from tvm._ffi.function import _init_api

_init_api("relay._expr", __name__)
