"""FFI implementation details"""
from __future__ import absolute_import as _abs
from tvm._ffi.function import list_global_func_names, get_global_func, _get_api
import sys

def _init_api(module_name, prefix):
    """Initialize api for a given module name

    mod : str
       The name of the module.
    """
    module = sys.modules[module_name]

    for name in list_global_func_names():
        if not name.startswith(prefix):
            continue
        fname = name[len(prefix)+1:]
        target_module = module

        if fname.find(".") != -1:
            continue
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = ("TVM PackedFunc %s. " % fname)
        setattr(target_module, ff.__name__, ff)

