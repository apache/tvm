"""Namespace of internal API

The functions in this namespace are automatically exported from C++ side via PackedFunc
that is registered by "TVM_REGISTER_*" macro. This way makes calling Python functions from C++
side very easily.

Each string starts with "_" in the "TVM_REGISTER_*" macro is an internal API. You can find
all the functions in "api_lang.cc", "api_base.cc", "api_arith.cc" and "api_ir.cc" under "src/api".
"""
