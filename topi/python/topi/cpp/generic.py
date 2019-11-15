"""FFI for generic TOPI ops and schedules"""

from tvm._ffi.function import _init_api_prefix

_init_api_prefix("topi.cpp.generic", "topi.generic")
