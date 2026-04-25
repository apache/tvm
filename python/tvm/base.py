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

import ctypes
import os
import sys
from pathlib import Path

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


def _resolve_lib(target_name: str) -> Path | None:
    """Find ``lib<target_name>.{so,dylib,dll}`` under TVM's package paths."""
    if sys.platform.startswith("win32"):
        basenames = (f"{target_name}.dll",)
    elif sys.platform.startswith("darwin"):
        basenames = (f"lib{target_name}.dylib", f"lib{target_name}.so")
    else:
        basenames = (f"lib{target_name}.so",)
    for d in libinfo.package_lib_paths():
        for name in basenames:
            p = d / name
            if p.is_file():
                return p.resolve()
    return None


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
_runtime_path = _resolve_lib("tvm_runtime")
if _runtime_path is None:
    raise RuntimeError(
        "Cannot find libtvm_runtime; searched: "
        + ", ".join(str(p) for p in libinfo.package_lib_paths())
    )
# Windows requires explicit DLL search-path setup before CDLL.
if sys.platform.startswith("win32"):
    os.add_dll_directory(str(_runtime_path.parent))
_LIB_RUNTIME = ctypes.CDLL(str(_runtime_path), ctypes.RTLD_GLOBAL)

_RUNTIME_ONLY = os.environ.get("TVM_USE_RUNTIME_LIB", "0").lower() in ("1", "true", "yes")
if _RUNTIME_ONLY:
    _LIB = _LIB_RUNTIME
else:
    _compiler_path = _resolve_lib("tvm_compiler")
    if _compiler_path is None:
        # Compiler lib not present — fall back to runtime-only mode.
        _LIB = _LIB_RUNTIME
        _RUNTIME_ONLY = True
    else:
        if sys.platform.startswith("win32"):
            os.add_dll_directory(str(_compiler_path.parent))
        _LIB = ctypes.CDLL(str(_compiler_path), ctypes.RTLD_LOCAL)


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
