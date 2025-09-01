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
# specific language governing permissions and limitations.
# Base logic to load library for extension package
import tvm_ffi
import os
import sys


def _load_lib():
    # first look at the directory of the current file
    file_dir = os.path.dirname(os.path.realpath(__file__))

    if sys.platform.startswith("win32"):
        lib_dll_name = "my_ffi_extension.dll"
    elif sys.platform.startswith("darwin"):
        lib_dll_name = "my_ffi_extension.dylib"
    else:
        lib_dll_name = "my_ffi_extension.so"

    lib_path = os.path.join(file_dir, lib_dll_name)
    return tvm_ffi.load_module(lib_path)


_LIB = _load_lib()
