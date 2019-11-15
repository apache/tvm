"""FFI for vision TOPI ops and schedules"""

from tvm._ffi.function import _init_api_prefix

from . import yolo

_init_api_prefix("topi.cpp.vision", "topi.vision")
