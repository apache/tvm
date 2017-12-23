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
ivec_create = tvm.get_global_func("tvm_ext.ivec_create")
ivec_get = tvm.get_global_func("tvm_ext.ivec_get")

class IntVec(object):
    """Example for using extension class in c++ """
    _tvm_tcode = 17

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        # You can also call your own customized
        # deleter if you can free it via your own FFI.
        tvm.nd.free_extension_handle(self.handle, 17)

    @property
    def _tvm_handle(self):
        return self.handle.value

    def __getitem__(self, idx):
        return ivec_get(self, idx)

# Register IntVec extension on python side.
tvm.register_extension(IntVec, IntVec)
