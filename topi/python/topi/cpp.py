"""FFI for C++ TOPI ops and schedules"""
from ._ffi.cpp_init import _init_api

_init_api("topi.cpp", "topi")