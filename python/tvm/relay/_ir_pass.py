"""FFI exposing the Relay type inference and checking."""

from tvm._ffi.function import _init_api

_init_api("relay._ir_pass", __name__)
