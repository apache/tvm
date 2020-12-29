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

# pylint: disable=invalid-name, unused-import
"""Packed Function namespace."""
import ctypes
from tvm._ffi.base import _LIB, check_call, c_str, string_types, _FFI_MODE

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    from tvm._ffi._cy3.core import _set_class_packed_func, _set_class_module
    from tvm._ffi._cy3.core import PackedFuncBase
    from tvm._ffi._cy3.core import convert_to_tvm_func
except (RuntimeError, ImportError) as error:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "cython":
        raise error
    from tvm._ffi._ctypes.packed_func import _set_class_packed_func, _set_class_module
    from tvm._ffi._ctypes.packed_func import PackedFuncBase
    from tvm._ffi._ctypes.packed_func import convert_to_tvm_func


PackedFuncHandle = ctypes.c_void_p


class PackedFunc(PackedFuncBase):
    """The PackedFunc object used in TVM.

    Function plays an key role to bridge front and backend in TVM.
    Function provide a type-erased interface, you can call function with positional arguments.

    The compiled module returns Function.
    TVM backend also registers and exposes its API as Functions.

    The following are list of common usage scenario of tvm.runtime.PackedFunc.

    - Automatic exposure of C++ API into python
    - To call PackedFunc from python side
    - To call python callbacks to inspect results in generated code
    - Bring python hook into C++ backend

    See Also
    --------
    tvm.register_func: How to register global function.
    tvm.get_global_func: How to get global function.
    """


_set_class_packed_func(PackedFunc)
