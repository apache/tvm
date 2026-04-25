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

# Caller-supplied dev-mode search list for ``libinfo.load_lib_ctypes``.
# These cover the wheel-install layout (``tvm/lib/``) and the in-tree dev
# build layout (``<worktree>/build/lib`` and ``<worktree>/lib``). Caller
# dirs win precedence over the self-anchored fallback inside ``libinfo``.
# TODO: remove once tvm-ffi exposes ``extra_lib_paths`` upstream
# (tracked in tdev issue #63).
_TVM_PKG_ROOT = Path(__file__).parent  # python/tvm/
_EXTRA_LIB_PATHS = [
    _TVM_PKG_ROOT / "lib",  # wheel layout
    _TVM_PKG_ROOT.parent.parent / "build" / "lib",  # dev: <worktree>/build/lib
    _TVM_PKG_ROOT.parent.parent / "lib",  # dev: <worktree>/lib
]


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

    # ``libinfo.load_lib_ctypes`` mirrors tvm_ffi's loader. We pass an explicit
    # ``extra_lib_paths`` so dev mode (PYTHONPATH=<repo>/python) finds TVM's
    # own ``build/lib/libtvm_*.so`` rather than the tvm-ffi tree.
    runtime_lib = libinfo.load_lib_ctypes(
        "tvm", "tvm_runtime", "RTLD_GLOBAL", extra_lib_paths=_EXTRA_LIB_PATHS
    )

    if runtime_only:
        return runtime_lib, _runtime_basename()

    try:
        compiler_lib = libinfo.load_lib_ctypes(
            "tvm", "tvm_compiler", "RTLD_LOCAL", extra_lib_paths=_EXTRA_LIB_PATHS
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
