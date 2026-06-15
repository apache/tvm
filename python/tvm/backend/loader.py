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
"""Backend loading and public alias support."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from tvm_ffi.libinfo import load_lib_ctypes

_LOADED_BACKENDS: dict[str, Any] = {}
_RUNTIME_SIDECAR_LOAD_ATTEMPTED: set[str] = set()


class _AliasModule(types.ModuleType):
    """Module object that exposes a backend module under a public alias."""

    def __init__(self, fullname: str, module):
        super().__init__(fullname, getattr(module, "__doc__", None))
        self.__dict__["__tvm_backend_module__"] = module
        self.__dict__["__package__"] = fullname.rpartition(".")[0]
        if hasattr(module, "__all__"):
            self.__dict__["__all__"] = module.__all__
        if hasattr(module, "__path__"):
            self.__dict__["__path__"] = []

    def __getattr__(self, name: str):
        return getattr(self.__dict__["__tvm_backend_module__"], name)

    def __setattr__(self, name: str, value):
        setattr(self.__dict__["__tvm_backend_module__"], name, value)

    def __delattr__(self, name: str):
        delattr(self.__dict__["__tvm_backend_module__"], name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.__dict__["__tvm_backend_module__"])))


class _AliasLoader:
    """Loader that returns an already-resolved module for an alias spec."""

    def __init__(self, fullname: str, module):
        self._fullname = fullname
        self._module = module

    def create_module(self, spec):
        return _get_alias_module(self._fullname, self._module)

    def exec_module(self, module):
        _set_module_alias(self._fullname, self._module)
        return None

    def is_package(self, fullname):
        return hasattr(self._module, "__path__")


def _redirect_tirx_backend_alias(fullname: str) -> str | None:
    prefix = "tvm.tirx."
    if not fullname.startswith(prefix):
        return None
    rest = fullname[len(prefix) :]
    backend_name, sep, tail = rest.partition(".")
    if not sep or backend_name not in _LOADED_BACKENDS:
        return None
    return f"tvm.backend.{backend_name}.{tail}"


class _BackendAliasFinder:
    """Redirect ``tvm.tirx.<backend>.*`` imports to ``tvm.backend.<backend>.*``."""

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        redirected = _redirect_tirx_backend_alias(fullname)
        if redirected is None:
            return None
        module = importlib.import_module(redirected)
        _set_module_alias(fullname, module)
        loader = _AliasLoader(fullname, module)
        spec = importlib.util.spec_from_loader(
            fullname, loader, is_package=hasattr(module, "__path__")
        )
        if spec is not None and hasattr(module, "__path__"):
            spec.submodule_search_locations = []
        return spec


if not any(isinstance(finder, _BackendAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _BackendAliasFinder())


def _get_alias_module(alias: str, module):
    existing = sys.modules.get(alias)
    if (
        isinstance(existing, _AliasModule)
        and existing.__dict__.get("__tvm_backend_module__") is module
    ):
        return existing
    return _AliasModule(alias, module)


def _set_module_alias(alias: str, module, *, direct: bool = False) -> None:
    alias_module = module if direct else _get_alias_module(alias, module)
    sys.modules[alias] = alias_module
    parent_name, _, child_name = alias.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child_name, alias_module)


def _alias_loaded_backend_modules(name: str) -> None:
    backend_prefix = f"tvm.backend.{name}"
    public_prefix = f"tvm.tirx.{name}"
    for module_name, module in sorted(list(sys.modules.items())):
        if module_name == backend_prefix or module_name.startswith(f"{backend_prefix}."):
            public_name = f"{public_prefix}{module_name[len(backend_prefix) :]}"
            _set_module_alias(public_name, module, direct=module_name == backend_prefix)


def _import_backend(name: str):
    module_name = f"tvm.backend.{name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name == module_name:
            raise ImportError(
                f"Cannot load TVM backend {name!r}: expected Python package {module_name!r}. "
                "Install the backend package or check the backend name."
            ) from err
        raise


def _load_runtime_sidecar(name: str, loaded_libs: dict[str, Any] | None = None) -> None:
    """Load ``libtvm_runtime_<name>`` next to ``libtvm_runtime`` if present."""
    target_name = f"tvm_runtime_{name}"
    if target_name in _RUNTIME_SIDECAR_LOAD_ATTEMPTED:
        return

    if loaded_libs is None:
        from tvm.base import _LOADED_LIBS  # pylint: disable=import-outside-toplevel

        loaded_libs = _LOADED_LIBS

    runtime_lib = loaded_libs.get("tvm_runtime")
    if runtime_lib is None:
        return

    _RUNTIME_SIDECAR_LOAD_ATTEMPTED.add(target_name)
    runtime_dir = Path(runtime_lib._name).resolve().parent
    try:
        loaded_libs[target_name] = load_lib_ctypes(
            package="tvm",
            target_name=target_name,
            mode="RTLD_GLOBAL",
            extra_lib_paths=[runtime_dir],
        )
    except (OSError, FileNotFoundError, RuntimeError):
        pass


def load(name: str) -> None:
    """Load a backend's Python registration hooks.

    Loading is idempotent.  A backend package must live at ``tvm.backend.<name>``
    and expose ``register_backend()``.
    """

    if name in _LOADED_BACKENDS:
        return None

    module = _import_backend(name)
    register_backend = getattr(module, "register_backend", None)
    if register_backend is None:
        raise AttributeError(f"Backend package 'tvm.backend.{name}' has no register_backend()")

    import tvm.tirx as tirx  # pylint: disable=import-outside-toplevel

    setattr(tirx, name, module)
    sys.modules[f"tvm.tirx.{name}"] = module
    _LOADED_BACKENDS[name] = module
    try:
        register_backend()
        _alias_loaded_backend_modules(name)
    except Exception:
        _LOADED_BACKENDS.pop(name, None)
        if getattr(tirx, name, None) is module:
            delattr(tirx, name)
        if sys.modules.get(f"tvm.tirx.{name}") is module:
            sys.modules.pop(f"tvm.tirx.{name}", None)
        raise
    return None


def is_loaded(name: str) -> bool:
    """Return whether a backend has been loaded."""

    return name in _LOADED_BACKENDS


__all__ = ["is_loaded", "load"]
