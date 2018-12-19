"""FFI exposing automatic differentiation"""

from tvm._ffi.function import _init_api

_init_api("relay._gradient", __name__)
