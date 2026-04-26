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
# pylint: disable=invalid-name, import-outside-toplevel
# ruff: noqa: F401
"""Base library for TVM."""

import os
import sys

from . import libinfo

# ----------------------------
# Python3 version.
# ----------------------------
if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 9):
    PY3STATEMENT = "The minimal Python requirement is Python 3.9"
    raise Exception(PY3STATEMENT)

# ----------------------------
# library loading
# ----------------------------

# The TVM C++ side is split into two shared libraries:
#
# - ``libtvm_runtime`` — runtime-only sources. Loaded with ``RTLD_GLOBAL`` so
#   its symbols are exposed to subsequent loads (NVRTC kernels, downstream
#   modules and so on resolve runtime symbols at link time).
# - ``libtvm_compiler`` — compiler / IR / transform sources, links against
#   ``libtvm_runtime``. Loaded with ``RTLD_LOCAL`` so compiler internals
#   don't leak into the global symbol namespace.
#
# If the environment variable ``TVM_USE_RUNTIME_LIB`` is truthy, or the
# compiler library is simply not present (runtime-only wheel), only the
# runtime is loaded and ``_LIB`` aliases ``_LIB_RUNTIME``.
_extra_lib_paths = libinfo.package_lib_paths()
_LIB_RUNTIME = libinfo.load_lib_ctypes(
    "tvm", "tvm_runtime", "RTLD_GLOBAL", extra_lib_paths=_extra_lib_paths
)

_RUNTIME_ONLY = libinfo.use_runtime_lib()
if _RUNTIME_ONLY:
    _LIB = _LIB_RUNTIME
else:
    try:
        _LIB = libinfo.load_lib_ctypes(
            "tvm", "tvm_compiler", "RTLD_LOCAL", extra_lib_paths=_extra_lib_paths
        )
    except RuntimeError:
        # Compiler lib not present — fall back to runtime-only mode.
        _LIB = _LIB_RUNTIME
        _RUNTIME_ONLY = True


try:
    # The following import is needed for TVM to work with pdb
    import readline  # pylint: disable=unused-import
except ImportError:
    pass

# version number
__version__ = libinfo.__version__


if _RUNTIME_ONLY:
    from tvm_ffi import registry as _tvm_ffi_registry

    _tvm_ffi_registry._SKIP_UNKNOWN_OBJECTS = True

# The FFI mode of TVM
_FFI_MODE = os.environ.get("TVM_FFI", "auto")

if _FFI_MODE == "ctypes":
    raise ImportError("We have phased out ctypes support in favor of cython on wards")


def py_str(x):
    return x.decode("utf-8")


TVMError = Exception
