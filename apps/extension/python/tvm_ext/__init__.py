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
        os.path.join(curr_path, "../../lib/libtvm_ext.so"), ctypes.RTLD_GLOBAL)
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
        tvm.nd.free_extension_handle(self.handle, self.__class__._tvm_tcode)

    @property
    def _tvm_handle(self):
        return self.handle.value

    def __getitem__(self, idx):
        return ivec_get(self, idx)

# Register IntVec extension on python side.
tvm.register_extension(IntVec, IntVec)


nd_create = tvm.get_global_func("tvm_ext.nd_create")
nd_add_two = tvm.get_global_func("tvm_ext.nd_add_two")
nd_get_addtional_info = tvm.get_global_func("tvm_ext.nd_get_addtional_info")

class NDSubClass(tvm.nd.NDArrayBase):
    """Example for subclassing TVM's NDArray infrastructure.

    By inheriting TMV's NDArray, external libraries could
    leverage TVM's FFI without any modification.
    """
    # Should be consistent with the type-trait set in the backend
    _array_type_code = 1

    @staticmethod
    def create(addtional_info):
        return nd_create(addtional_info)

    @property
    def addtional_info(self):
        return nd_get_addtional_info(self)

    def __add__(self, other):
        return nd_add_two(self, other)

tvm.register_extension(NDSubClass, NDSubClass)
