"""Namespace of IR pass functions.

This namespace is used for developers. While you do not see any declarations.
The functions are automatically exported from C++ side via PackedFunc.

Each api is a PackedFunc that can be called in a positional argument manner.
You can read "include/tvm/pass.h" for the function signature of these functions.
"""
from ._ffi.function import _init_api

_init_api("tvm.ir_pass")
