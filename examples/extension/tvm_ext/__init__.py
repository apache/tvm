"""Example extension package of TVM."""
from __future__ import absolute_import
import os
import ctypes

def load_lib():
    """Load library, the functions will be registered into TVM"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib = ctypes.CDLL(os.path.join(curr_path, "../lib/libtvm_ext.so"),
                      ctypes.RTLD_GLOBAL)
    return lib

_LIB = load_lib()

import tvm
# Expose two functions into python
bind_add = tvm.get_global_func("tvm_ext.bind_add")
sym_add = tvm.get_global_func("tvm_ext.sym_add")

