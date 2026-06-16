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
"""Metal-owned TIRx modules."""

from importlib import import_module
from pathlib import Path

from tvm_ffi.libinfo import load_lib_ctypes

from tvm.base import _LOADED_LIBS

_LAZY_SUBMODULES = {"op", "script", "target_tags"}


def _detect_target_from_device(dev):
    from tvm.target import Target  # pylint: disable=import-outside-toplevel

    return Target(
        {
            "kind": "metal",
            "max_shared_memory_per_block": 32768,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def register_backend():
    """Register Metal-owned Python semantics."""
    from tvm.target.detect_target import register_device_target_detector
    from tvm.tirx.script.builder import ir as builder_ir  # pylint: disable=import-outside-toplevel

    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    try:
        # Runtime sidecars only need registration side effects; libtvm_runtime is global.
        _LOADED_LIBS["tvm_runtime_metal"] = load_lib_ctypes(
            package="tvm",
            target_name="tvm_runtime_metal",
            extra_lib_paths=[runtime_dir],
            mode="RTLD_LOCAL",
        )
    except (OSError, FileNotFoundError, RuntimeError):
        pass
    register_device_target_detector("metal", _detect_target_from_device)
    for name, namespace in script_namespaces().items():
        builder_ir.register_script_namespace(name, namespace)
    import_module(f"{__name__}.target_tags")


def script_namespaces(**_):
    """Return Metal-owned TVMScript namespaces."""
    from .script import MetalNamespace  # pylint: disable=import-outside-toplevel

    return {"metal": MetalNamespace()}


def script_namespace(**kwargs):
    """Return the Metal TVMScript namespace object."""
    return script_namespaces(**kwargs)["metal"]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "op",
    "register_backend",
    "script",
    "script_namespace",
    "script_namespaces",
    "target_tags",
]
