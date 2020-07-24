import tvm._ffi

from tvm.runtime import Object
from . import _ffi_api

@tvm._ffi.register_object("meta.VarDef")
class VarDef(Object):
    def __init__(self, name, type_info):
        self.__init_handle_by_constructor__(
            _ffi_api.VarDef, name, type_info)

@tvm._ffi.register_object("meta.ObjectDef")
class ObjectDef(Object):
    def __init__(self, name, ref_name, nmspace, base=None, variables=[]):
        self.__init_handle_by_constructor__(
            _ffi_api.ObjectDef, name, ref_name, nmspace, base, variables)
