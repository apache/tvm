"""FFI for image TOPI ops and schedules"""

from tvm._ffi.function import _init_api_prefix

_init_api_prefix("topi.cpp.image", "topi.image")
