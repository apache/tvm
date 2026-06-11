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
"""Base library for TVM."""

import os
from pathlib import Path

from tvm_ffi.libinfo import load_lib_ctypes

from . import libinfo

# ----------------------------
# library loading
# ----------------------------

# Whether only the runtime library is loaded (runtime-only wheel, or
# ``TVM_USE_RUNTIME_LIB=1``). Set during library loading below.
_RUNTIME_ONLY = os.environ.get("TVM_USE_RUNTIME_LIB") == "1"

# Handles of the core libraries actually loaded, keyed by basename
# (e.g. ``{"tvm_runtime": <CDLL>, "tvm_compiler": <CDLL>}``). Downstream /
# autoloaded extensions can inspect this to skip duplicate libraries
# (``"tvm_runtime" in _LOADED_LIBS``) and obtain the loaded handle.
_LOADED_LIBS = {}


def load_backend_libs(runtime_lib_path: str) -> None:
    """Load each known backend runtime DSO into ``_LOADED_LIBS``; failures are silent."""
    # Known per-backend runtime DSOs that, when present, are loaded with
    # RTLD_GLOBAL so their static initializers register the device backend.
    backend_runtime_libs = ["cuda", "vulkan", "opencl", "metal", "rocm", "hexagon", "extra"]
    runtime_dir = Path(runtime_lib_path).resolve().parent
    for backend in backend_runtime_libs:
        target_name = f"tvm_runtime_{backend}"
        try:
            _LOADED_LIBS[target_name] = load_lib_ctypes(
                package="tvm",
                target_name=target_name,
                mode="RTLD_GLOBAL",
                extra_lib_paths=[runtime_dir],
            )
        except (OSError, FileNotFoundError, RuntimeError):
            pass


# runtime is loaded RTLD_GLOBAL to expose its symbols to subsequent loads;
# compiler is loaded RTLD_LOCAL.
_LOADED_LIBS["tvm_runtime"] = load_lib_ctypes(
    "tvm", "tvm_runtime", "RTLD_GLOBAL", extra_lib_paths=libinfo.package_lib_paths()
)

# After libtvm_runtime.so is in the global symbol namespace, scan the same
# directory for per-backend DSOs (libtvm_runtime_cuda.so, etc.) and load each
# with RTLD_GLOBAL so their static initializers register device backends.
load_backend_libs(_LOADED_LIBS["tvm_runtime"]._name)

if not _RUNTIME_ONLY:
    try:
        _LOADED_LIBS["tvm_compiler"] = load_lib_ctypes(
            "tvm", "tvm_compiler", "RTLD_LOCAL", extra_lib_paths=libinfo.package_lib_paths()
        )
    except (RuntimeError, OSError):
        # Compiler lib not present, or present but unloadable (missing LLVM
        # deps / linker issues) — fall back to runtime-only mode.
        _RUNTIME_ONLY = True

if _RUNTIME_ONLY:
    from tvm_ffi import registry as _tvm_ffi_registry

    _tvm_ffi_registry._SKIP_UNKNOWN_OBJECTS = True
