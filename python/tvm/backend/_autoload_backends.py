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
"""Autoload backend libraries and Python backend registration hooks."""

from __future__ import annotations

import os
import warnings
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from tvm_ffi.libinfo import load_lib_ctypes

_BUILTIN_BACKENDS = (
    "cuda",
    "metal",
    "rocm",
    "trn",
    "opencl",
    "vulkan",
    "webgpu",
    "hexagon",
    "adreno",
)
_LEGACY_RUNTIME_LIBS_WITHOUT_BACKEND_PACKAGE = ("extra",)

# Guard so autoload runs at most once per process, even if invoked again.
_BACKEND_LIBS_LOADED = False
_AUTO_LOAD_DONE = False


def autoload_backend_libs(loaded_libs: dict[str, Any] | None = None) -> None:
    """Load each known backend runtime DSO into the process-global symbol namespace."""
    global _BACKEND_LIBS_LOADED
    if _BACKEND_LIBS_LOADED:
        return

    if loaded_libs is None:
        from tvm.base import _LOADED_LIBS  # pylint: disable=import-outside-toplevel

        loaded_libs = _LOADED_LIBS

    runtime_lib = loaded_libs.get("tvm_runtime")
    if runtime_lib is None:
        return

    _BACKEND_LIBS_LOADED = True

    runtime_dir = Path(runtime_lib._name).resolve().parent
    for runtime_lib_name in _backend_runtime_lib_names():
        target_name = f"tvm_runtime_{runtime_lib_name}"
        try:
            loaded_libs[target_name] = load_lib_ctypes(
                package="tvm",
                target_name=target_name,
                mode="RTLD_GLOBAL",
                extra_lib_paths=[runtime_dir],
            )
        except (OSError, FileNotFoundError, RuntimeError):
            pass


def _backend_runtime_lib_names() -> tuple[str, ...]:
    runtime_libs = []
    for backend in _BUILTIN_BACKENDS:
        module = import_module(f"tvm.backend.{backend}")
        runtime_libs.extend(getattr(module, "RUNTIME_LIBS", ()))
    runtime_libs.extend(_LEGACY_RUNTIME_LIBS_WITHOUT_BACKEND_PACKAGE)
    return tuple(runtime_libs)


def load_all() -> None:
    """Load all in-tree backend Python hooks."""
    from . import load  # pylint: disable=import-outside-toplevel

    for name in _BUILTIN_BACKENDS:
        load(name)
    return None


def _autoload_backends() -> None:
    """Discover and invoke out-of-tree backends registered via entry points."""
    global _AUTO_LOAD_DONE
    if _AUTO_LOAD_DONE:
        return
    _AUTO_LOAD_DONE = True

    if os.environ.get("TVM_DEVICE_BACKEND_AUTOLOAD", "1") == "0":
        return

    for entry_pt in entry_points(group="tvm.backends"):
        try:
            entry_pt.load()()
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(f"Failed to autoload tvm backend '{entry_pt.name}': {e}")
