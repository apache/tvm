"""Example extension package of TVM."""
from __future__ import absolute_import
import os
import ctypes
# Import TVM first to get library symbols
import tvm

def load_lib():
    """Load library, the functions will be registered into TVM"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # load in as global so the global extern symbol is visible to other dll.
    lib = ctypes.CDLL(
        os.path.join(curr_path, "../../lib/libtvm_ext_json_rt.so"), ctypes.RTLD_GLOBAL)
    return lib

_LIB = load_lib()

create_json_rt = tvm.get_global_func("ext_json_rt.create_json_rt")

