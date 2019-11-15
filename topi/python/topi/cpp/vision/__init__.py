"""FFI for vision TOPI ops and schedules"""

from . import yolo
from tvm._ffi.function import _init_api_prefix

_init_api_prefix("topi.cpp.vision", "topi.vision")
