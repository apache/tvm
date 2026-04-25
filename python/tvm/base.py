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
import importlib.metadata as _im
import os
import sys

import tvm_ffi

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


def _load_lib():
    """Load TVM C++ libraries.

    The TVM C++ side is split into two shared libraries:

    - ``libtvm_runtime`` — runtime-only sources. Loaded with ``RTLD_GLOBAL`` so
      its symbols are exposed to subsequent loads (NVRTC kernels, downstream
      modules and so on resolve runtime symbols at link time).
    - ``libtvm_compiler`` — compiler / IR / transform sources, links against
      ``libtvm_runtime``. Loaded with ``RTLD_LOCAL`` so compiler internals
      don't leak into the global symbol namespace.

    If the environment variable ``TVM_USE_RUNTIME_LIB`` is set (truthy), only
    the runtime is loaded — useful for runtime-only deployments where the
    compiler library is not shipped.

    Returns
    -------
    tuple of (loaded_lib, basename)
        The handle returned is the compiler library (when loaded), otherwise
        the runtime library. The basename is used by callers to determine
        whether they're in runtime-only mode (``"runtime" in basename``).
    """
    # The dll search path need to be added explicitly in windows
    if sys.platform.startswith("win32"):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)

    runtime_only = bool(os.environ.get("TVM_USE_RUNTIME_LIB", False))

    # Preferred path: tvm_ffi.libinfo.load_lib_ctypes resolves shared libs via
    # the wheel RECORD or a dev-mode `lib/`/`build/lib/` fallback.
    try:
        runtime_lib = tvm_ffi.libinfo.load_lib_ctypes(
            package="tvm", target_name="tvm_runtime", mode="RTLD_GLOBAL"
        )
    except (RuntimeError, _im.PackageNotFoundError):
        return _load_lib_dev_fallback(runtime_only=runtime_only)

    if runtime_only:
        return runtime_lib, _runtime_basename()

    try:
        compiler_lib = tvm_ffi.libinfo.load_lib_ctypes(
            package="tvm", target_name="tvm_compiler", mode="RTLD_LOCAL"
        )
        return compiler_lib, _compiler_basename()
    except RuntimeError:
        # Compiler lib is not present — wheel ships only the runtime, or the
        # user is in a runtime-only environment.
        return runtime_lib, _runtime_basename()


def _runtime_basename() -> str:
    if sys.platform.startswith("win32"):
        return "tvm_runtime.dll"
    if sys.platform.startswith("darwin"):
        return "libtvm_runtime.dylib"
    return "libtvm_runtime.so"


def _compiler_basename() -> str:
    if sys.platform.startswith("win32"):
        return "tvm_compiler.dll"
    if sys.platform.startswith("darwin"):
        return "libtvm_compiler.dylib"
    return "libtvm_compiler.so"


def _load_lib_dev_fallback(runtime_only: bool):
    """PYTHONPATH / dev-build fallback when load_lib_ctypes can't find package metadata."""
    runtime_paths = libinfo.find_lib_path(name=_runtime_basename(), optional=False)
    runtime_lib = ctypes.CDLL(runtime_paths[0], ctypes.RTLD_GLOBAL)
    if runtime_only:
        return runtime_lib, os.path.basename(runtime_paths[0])
    compiler_paths = libinfo.find_lib_path(name=_compiler_basename(), optional=True)
    if compiler_paths:
        # NOTE: compiler is RTLD_LOCAL — internals must not leak globally.
        compiler_lib = ctypes.CDLL(compiler_paths[0], ctypes.RTLD_LOCAL)
        return compiler_lib, os.path.basename(compiler_paths[0])
    return runtime_lib, os.path.basename(runtime_paths[0])


try:
    # The following import is needed for TVM to work with pdb
    import readline  # pylint: disable=unused-import
except ImportError:
    pass

# version number
__version__ = libinfo.__version__
# library instance
_LIB, _LIB_NAME = _load_lib()

# Whether we are runtime only
_RUNTIME_ONLY = "runtime" in _LIB_NAME


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
