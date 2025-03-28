# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, unused-import, wrong-import-position
"""FFI registry to register function and objects."""
import sys
import ctypes

from .base import _LIB, check_call, py_str, c_str, string_types, _RUNTIME_ONLY

from ._cy3.core import _register_object, _get_object_type_index
from ._cy3.core import _reg_extension
from ._cy3.core import convert_to_tvm_func, _get_global_func, PackedFuncBase


def register_object(type_key=None):
    """register object type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node

    Examples
    --------
    The following code registers MyObject
    using type key "test.MyObject"

    .. code-block:: python

      @tvm.register_object("test.MyObject")
      class MyObject(Object):
          pass
    """
    object_name = type_key if isinstance(type_key, str) else type_key.__name__

    def register(cls):
        """internal register function"""
        if hasattr(cls, "_type_index"):
            tindex = cls._type_index
        else:
            tidx = ctypes.c_uint()
            if not _RUNTIME_ONLY:
                check_call(_LIB.TVMObjectTypeKey2Index(c_str(object_name), ctypes.byref(tidx)))
            else:
                # directly skip unknown objects during runtime.
                ret = _LIB.TVMObjectTypeKey2Index(c_str(object_name), ctypes.byref(tidx))
                if ret != 0:
                    return cls
            tindex = tidx.value
        _register_object(tindex, cls)
        return cls

    if isinstance(type_key, str):
        return register

    return register(type_key)


def get_object_type_index(cls):
    """
    Get type index of object type

    Parameters
    ----------
    cls : type
        The object type to get type index for.

    Returns
    -------
    type_index : Optional[int]
        The type index, or None if type not found in the registry.
    """
    return _get_object_type_index(cls)


def register_func(func_name, f=None, override=False):
    """Register global function

    Parameters
    ----------
    func_name : str or function
        The function name

    f : function, optional
        The function to be registered.

    override: boolean optional
        Whether override existing entry.

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    The following code registers my_packed_func as global function.
    Note that we simply get it back from global function table to invoke
    it from python side. However, we can also invoke the same function
    from C++ backend, or in the compiled TVM code.

    .. code-block:: python

      targs = (10, 10.0, "hello")
      @tvm.register_func
      def my_packed_func(*args):
          assert(tuple(args) == targs)
          return 10
      # Get it out from global function table
      f = tvm.get_global_func("my_packed_func")
      assert isinstance(f, tvm.PackedFunc)
      y = f(*targs)
      assert y == 10
    """
    if callable(func_name):
        f = func_name
        func_name = f.__name__

    if not isinstance(func_name, str):
        raise ValueError("expect string function name")

    ioverride = ctypes.c_int(override)

    def register(myf):
        """internal register function"""
        if not isinstance(myf, PackedFuncBase):
            myf = convert_to_tvm_func(myf)
        check_call(_LIB.TVMFuncRegisterGlobal(c_str(func_name), myf.handle, ioverride))
        return myf

    if f:
        return register(f)
    return register


def get_global_func(name, allow_missing=False):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    allow_missing : bool
        Whether allow missing function or raise an error.

    Returns
    -------
    func : PackedFunc
        The function to be returned, None if function is missing.
    """
    return _get_global_func(name, allow_missing)


def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.TVMFuncListGlobalNames(ctypes.byref(size), ctypes.byref(plist)))
    fnames = []
    for i in range(size.value):
        fnames.append(py_str(plist[i]))
    return fnames


def extract_ext_funcs(finit):
    """
    Extract the extension PackedFuncs from a C module.

    Parameters
    ----------
    finit : ctypes function
        a ctypes that takes signature of TVMExtensionDeclarer

    Returns
    -------
    fdict : dict of str to Function
        The extracted functions
    """
    fdict = {}

    def _list(name, func):
        fdict[name] = func

    myf = convert_to_tvm_func(_list)
    ret = finit(myf.handle)
    _ = myf
    if ret != 0:
        raise RuntimeError("cannot initialize with %s" % finit)
    return fdict


def remove_global_func(name):
    """Remove a global function by name

    Parameters
    ----------
    name : str
        The name of the global function
    """
    check_call(_LIB.TVMFuncRemoveGlobal(c_str(name)))


def _get_api(f):
    flocal = f
    flocal.is_global = True
    return flocal


def _init_api(namespace, target_module_name=None):
    """Initialize api for a given module name

    namespace : str
       The namespace of the source registry

    target_module_name : str
       The target module name if different from namespace
    """
    target_module_name = target_module_name if target_module_name else namespace
    if namespace.startswith("tvm."):
        _init_api_prefix(target_module_name, namespace[4:])
    else:
        _init_api_prefix(target_module_name, namespace)


def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]

    for name in list_global_func_names():
        if not name.startswith(prefix):
            continue

        fname = name[len(prefix) + 1 :]
        target_module = module

        if fname.find(".") != -1:
            continue
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = "TVM PackedFunc %s. " % fname
        setattr(target_module, ff.__name__, ff)
