"""namespace of IR node builder make function

This namespace is used for developers. While you do not see any declarations.
The functions are automatically exported from C++ side via PackedFunc.

Each api is a PackedFunc that can be called in a positional argument manner.
You can use make function to build the IR node.
"""
from ._ffi.function import _init_api

_init_api("tvm.make")
