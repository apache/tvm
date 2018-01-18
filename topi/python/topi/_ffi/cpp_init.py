"""FFI implementation details"""
from __future__ import absolute_import as _abs
import sys
from tvm._ffi.function import list_global_func_names, get_global_func, _get_api

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
        func = _get_api(f)
        func.__name__ = fname
        func.__doc__ = ("TVM PackedFunc %s. " % fname)
        setattr(target_module, func.__name__, func)
