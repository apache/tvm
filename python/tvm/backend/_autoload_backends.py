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
"""Autoload built-in and out-of-tree backend libraries and registration hooks."""

from __future__ import annotations

import os
import warnings
from importlib.metadata import entry_points

from tvm.backend.loader import _load_runtime_lib, load

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

_AUTO_LOAD_DONE = False


def _load_builtin_backends() -> None:
    """Load all in-tree backend Python hooks."""
    for name in _BUILTIN_BACKENDS:
        load(name)
    return None


def _autoload_backends() -> None:
    """Load built-in backends and invoke out-of-tree backend entry points."""
    global _AUTO_LOAD_DONE
    if _AUTO_LOAD_DONE:
        return
    _AUTO_LOAD_DONE = True

    if os.environ.get("TVM_DEVICE_BACKEND_AUTOLOAD", "1") == "0":
        return

    _load_runtime_lib("extra")

    from tvm import _RUNTIME_ONLY  # pylint: disable=import-outside-toplevel

    if not _RUNTIME_ONLY:
        _load_builtin_backends()

    # Out-of-tree extensions opt into being loaded automatically at ``import tvm`` time
    # by declaring an entry point in the ``tvm.backends`` group:
    # [project.entry-points."tvm.backends"] tvm_foo = "tvm_foo:_autoload".
    # Autoload can be disabled via ``TVM_DEVICE_BACKEND_AUTOLOAD=0``.
    for entry_pt in entry_points(group="tvm.backends"):
        try:
            entry_pt.load()()
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(f"Failed to autoload tvm backend '{entry_pt.name}': {e}")
