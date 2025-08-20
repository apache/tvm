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
# coding: utf-8
"""Base library for TVM FFI."""
import ctypes
import os
import sys
import subprocess
import logging
from . import libinfo

logger = logging.getLogger(__name__)

# ----------------------------
# Python3 version.
# ----------------------------
if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 9):
    PY3STATEMENT = "The minimal Python requirement is Python 3.9"
    raise Exception(PY3STATEMENT)

# ----------------------------
# library loading
# ----------------------------


def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_libtvm_ffi()
    # The dll search path need to be added explicitly in windows
    if sys.platform.startswith("win32"):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)

    lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
    return lib


# library instance
_LIB = _load_lib()
