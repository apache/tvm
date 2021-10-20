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
"""Support infra of TVM."""
import ctypes
import tvm._ffi
from .runtime.module import Module
from . import get_global_func


def libinfo():
    """Returns a dictionary containing compile-time info, including cmake flags and git commit hash

    Returns
    -------
    info: Dict[str, str]
        The dictionary of compile-time info.
    """
    get_lib_info_func = get_global_func("support.GetLibInfo", allow_missing=True)
    if get_lib_info_func is not None:
        lib_info = get_lib_info_func()
        if lib_info is None:
            return {}
    else:
        return {}
    return dict(lib_info.items())


class FrontendTestModule(Module):
    """A tvm.runtime.Module whose member functions are PackedFunc."""

    def __init__(self, entry_name=None):
        underlying_mod = get_global_func("testing.FrontendTestModule")()
        handle = underlying_mod.handle

        # Set handle to NULL to avoid cleanup in c++ runtime, transferring ownership.
        # Both cython and ctypes FFI use c_void_p, so this is safe to assign here.
        underlying_mod.handle = ctypes.c_void_p(0)

        super(FrontendTestModule, self).__init__(handle)
        if entry_name is not None:
            self.entry_name = entry_name

    def add_function(self, name, func):
        self.get_function("__add_function")(name, func)

    def __setitem__(self, key, value):
        self.add_function(key, value)


tvm._ffi._init_api("support", __name__)
