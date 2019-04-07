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
"""FFI for C++ TOPI ops and schedules"""
import sys
import os
import ctypes
from imp import new_module as _new_module
from tvm._ffi.function import _init_api_prefix
from tvm._ffi import libinfo

def _get_lib_names():
    if sys.platform.startswith('win32'):
        return ['libtvm_topi.dll', 'tvm_topi.dll']
    if sys.platform.startswith('darwin'):
        return ['libtvm_topi.dylib', 'tvm_topi.dylib']
    return ['libtvm_topi.so', 'tvm_topi.so']

def _load_lib():
    """Load libary by searching possible path."""
    curr_path = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    lib_search = curr_path
    lib_path = libinfo.find_lib_path(_get_lib_names(), lib_search, optional=True)
    if lib_path is None:
        return None, None
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    return lib, os.path.basename(lib_path[0])

_LIB, _LIB_NAME = _load_lib()

_init_api_prefix("topi.cpp", "topi")

def _create_module(name):
    fullname = __name__ + "." + name
    mod = _new_module(fullname)
    sys.modules[fullname] = mod
    return mod

# pylint: disable-msg=C0103
nn = _create_module("nn")
_init_api_prefix("topi.cpp.nn", "topi.nn")
generic = _create_module("generic")
_init_api_prefix("topi.cpp.generic", "topi.generic")
cuda = _create_module("cuda")
_init_api_prefix("topi.cpp.cuda", "topi.cuda")
rocm = _create_module("rocm")
_init_api_prefix("topi.cpp.rocm", "topi.rocm")
x86 = _create_module("x86")
_init_api_prefix("topi.cpp.x86", "topi.x86")
vision = _create_module("vision")
_init_api_prefix("topi.cpp.vision", "topi.vision")
yolo = _create_module("vision.yolo")
_init_api_prefix("topi.cpp.vision.yolo", "topi.vision.yolo")
image = _create_module("image")
_init_api_prefix("topi.cpp.image", "topi.image")
