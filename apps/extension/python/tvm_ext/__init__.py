"""Example extension package of TVM."""
from __future__ import absolute_import
import os
import ctypes
# Import TVM first to get library symbols
import tvm

def load_lib():
    """Load library, the functions will be registered into TVM"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib = ctypes.CDLL(os.path.join(curr_path, "../../lib/libtvm_ext.so"))
    return lib

_LIB = load_lib()

# Expose two functions into python
bind_add = tvm.get_global_func("tvm_ext.bind_add")
sym_add = tvm.get_global_func("tvm_ext.sym_add")

