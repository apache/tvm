"""FFI for x86 TOPI ops and schedules"""

from tvm._ffi.function import _init_api_prefix

_init_api_prefix("topi.cpp.x86", "topi.x86")
