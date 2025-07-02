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
"""FFI registry to register function and objects."""
import sys
from . import core

# whether we simplify skip unknown objects regtistration
_SKIP_UNKNOWN_OBJECTS = False


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

      @tvm.ffi.register_object("test.MyObject")
      class MyObject(Object):
          pass
    """
    object_name = type_key if isinstance(type_key, str) else type_key.__name__

    def register(cls):
        """internal register function"""
        type_index = core._object_type_key_to_index(object_name)
        if type_index is None:
            if _SKIP_UNKNOWN_OBJECTS:
                return cls
            raise ValueError("Cannot find object type index for %s" % object_name)
        core._add_class_attrs_by_reflection(type_index, cls)
        core._register_object_by_index(type_index, cls)
        return cls

    if isinstance(type_key, str):
        return register

    return register(type_key)


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
    """
    if callable(func_name):
        f = func_name
        func_name = f.__name__

    if not isinstance(func_name, str):
        raise ValueError("expect string function name")

    def register(myf):
        """internal register function"""
        return core._register_global_func(func_name, myf, override)

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
    func : Function
        The function to be returned, None if function is missing.
    """
    return core._get_global_func(name, allow_missing)


def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    name_functor = get_global_func("ffi.FunctionListGlobalNamesFunctor")()
    num_names = name_functor(-1)
    return [name_functor(i) for i in range(num_names)]


def remove_global_func(name):
    """Remove a global function by name

    Parameters
    ----------
    name : str
        The name of the global function
    """
    get_global_func("ffi.FunctionRemoveGlobal")(name)


def _init_api(namespace, target_module_name=None):
    """Initialize api for a given module name

    namespace : str
       The namespace of the source registry

    target_module_name : str
       The target module name if different from namespace
    """
    target_module_name = target_module_name if target_module_name else namespace

    if namespace.startswith("tvm."):
        prefix = namespace[4:]
    else:
        prefix = namespace

    target_module = sys.modules[target_module_name]

    for name in list_global_func_names():
        if not name.startswith(prefix):
            continue

        fname = name[len(prefix) + 1 :]
        if fname.find(".") != -1:
            continue

        f = get_global_func(name)
        f.__name__ = fname
        setattr(target_module, f.__name__, f)


__all__ = [
    "register_object",
    "register_func",
    "get_global_func",
    "list_global_func_names",
    "remove_global_func",
    "_init_api",
]
