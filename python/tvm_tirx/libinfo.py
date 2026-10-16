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
"""Library and include path information for TVM."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


def _rel_top_directory() -> Path:
    """Top directory in the installed (wheel) layout: ``python/tvm_tirx/``."""
    return Path(__file__).parent


def _dev_top_directory() -> Path:
    """Top directory in the source/dev layout: the worktree root."""
    return _rel_top_directory() / ".." / ".."


def package_lib_paths() -> list[Path]:
    """Return search directories for TVM's shared libraries.

    Anchored on this file's location (``python/tvm_tirx/libinfo.py``), the list
    covers the wheel-install layout (``python/tvm_tirx/lib/``) and the in-tree dev
    build layouts (``<worktree>/build/lib/`` and ``<worktree>/lib/``).
    ``TVM_TIRX_LIBRARY_PATH`` is prepended when set so it takes priority. Callers
    pick the basenames they want (e.g. ``libtvm_runtime.so``) and the load
    mode; this function only returns the search path.
    """
    pkg = _rel_top_directory()  # python/tvm_tirx/
    paths: list[Path] = []
    if os.environ.get("TVM_TIRX_LIBRARY_PATH"):
        paths.append(Path(os.environ["TVM_TIRX_LIBRARY_PATH"]))
    paths += [
        pkg / "lib",  # wheel layout
        _dev_top_directory() / "build" / "lib",  # dev: <worktree>/build/lib
        _dev_top_directory() / "lib",  # dev: <worktree>/lib
    ]
    dev = _dev_top_directory().resolve()
    if dev.name == "tirx" and (dev.parent / "CMakeLists.txt").is_file():
        paths.append(dev.parent / "build" / "lib")
    return paths


def find_libtvm_runtime() -> str:
    """Return the path to the ``libtvm_runtime`` shared library.

    Returns
    -------
    path : str
        The resolved path to the TVM runtime shared library.

    Raises
    ------
    RuntimeError
        If the runtime library cannot be found.
    """
    candidate = _ffi_info()._find_library_by_basename(
        "tvm_tirx", "tvm_runtime", extra_lib_paths=package_lib_paths()
    )
    if ret := _ffi_info()._resolve_and_validate([candidate], cond=lambda _: True):
        return ret
    raise RuntimeError("Cannot find libtvm_runtime")


def find_include_path() -> str:
    """Return the path to TVM's own ``include/`` directory."""
    if ret := _ffi_info()._resolve_and_validate(
        paths=[
            _rel_top_directory() / "include",
            _dev_top_directory() / "include",
        ],
        cond=lambda p: (p / "tvm" / "runtime").is_dir(),
    ):
        return ret
    raise RuntimeError("Cannot find TVM include path.")


__version__ = "0.1.0a1"


_LIBRARY_NAMES = ("tvm_tirx_compiler", "tvm_runtime")
_COMPONENTS = {
    "compiler": "tvm_tirx_compiler",
    "runtime": "tvm_runtime",
    "cuda": "tvm_runtime_cuda",
    "metal": "tvm_runtime_metal",
    "opencl": "tvm_runtime_opencl",
    "vulkan": "tvm_runtime_vulkan",
    "rocm": "tvm_runtime_rocm",
}


def _find_prefix(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "include/tvm/runtime/base.h").is_file():
            return candidate
    raise RuntimeError(f"Cannot find TIRx public headers above {start}")


def _paths(prefix: Path | None, library_dir: Path | None) -> tuple[Path, Path]:
    explicit_prefix = prefix is not None
    if library_dir is not None:
        library_dir = library_dir.resolve()
        prefix = prefix.resolve() if prefix is not None else _find_prefix(library_dir)
    else:
        prefix = prefix.resolve() if prefix is not None else _find_prefix(Path(__file__).parent)
        library_dir = prefix / "lib"
        candidates = [prefix / "lib", prefix / "lib64", prefix / "build/lib"]
        if prefix.name == "tirx":
            candidates.append(prefix.parent / "build/lib")
        if not explicit_prefix and os.environ.get("TVM_TIRX_LIBRARY_PATH"):
            candidates.insert(0, Path(os.environ["TVM_TIRX_LIBRARY_PATH"]))
        for candidate in candidates:
            if all(
                any(
                    (candidate / filename).is_file()
                    for filename in (f"lib{name}.so", f"lib{name}.dylib", f"{name}.dll")
                )
                for name in ("tvm_runtime",)
            ):
                library_dir = candidate
                break
    if not (prefix / "include/tvm/runtime/base.h").is_file():
        raise RuntimeError(f"Cannot find TIRx public headers in {prefix / 'include'}")
    return prefix, library_dir


def _libraries(library_dir: Path, platform: str, names=_LIBRARY_NAMES) -> dict[str, dict[str, str]]:
    libraries = {}
    windows = platform in ("Windows", "win32")
    apple = platform in ("Darwin", "iOS", "darwin")
    for name in names:
        if windows:
            filenames = (f"{name}.dll",)
        elif apple:
            filenames = (f"lib{name}.dylib", f"lib{name}.so")
        else:
            filenames = (f"lib{name}.so",)
        location = next((library_dir / f for f in filenames if (library_dir / f).is_file()), None)
        if location is None:
            raise RuntimeError(f"Cannot find TIRx library {name} in {library_dir}")
        implib = ""
        if windows:
            import_names = (f"{name}.lib", f"lib{name}.dll.a")
            import_library = next(
                (library_dir / f for f in import_names if (library_dir / f).is_file()), None
            )
            if import_library is None:
                raise RuntimeError(f"Missing import library for {location}")
            implib = str(import_library)
        libraries[name] = {"location": str(location), "implib": implib}
    return libraries


def configuration(
    prefix: Path | None = None,
    library_dir: Path | None = None,
    platform: str = sys.platform,
) -> dict:
    """Return the public include and core shared-library paths for one installation."""
    prefix, library_dir = _paths(prefix, library_dir)
    return {
        "version": __version__,
        "include_dir": str(prefix / "include"),
        "library_dir": str(library_dir),
        "cmake_dir": str(library_dir / "cmake/tvm_tirx"),
        "libraries": _libraries(library_dir, platform),
    }


def _ffi_info():
    """Read the selected FFI package's pure Python helper without importing its runtime."""
    spec = importlib.machinery.PathFinder.find_spec("tvm_ffi")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot find the matching apache-tvm-ffi Python package")
    helper = Path(spec.origin).parent / "libinfo.py"
    spec = importlib.util.spec_from_file_location("_tvm_ffi_libinfo", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _available(library_dir, platform):
    available = []
    for component, name in _COMPONENTS.items():
        try:
            _libraries(library_dir, platform, (name,))
        except RuntimeError:
            continue
        available.append(component)
    return available


def _link_inputs(library_dir, platform, components):
    available = _available(library_dir, platform)
    selected = list(dict.fromkeys(components or ["all"]))
    if "all" in selected:
        selected = [*available, *(name for name in selected if name != "all")]
    for name in selected:
        if name != "ffi" and name not in available:
            raise RuntimeError(
                f"Component '{name}' is unavailable; available: {' '.join(available)} ffi"
            )
    # Dependents precede their dependencies even if the command line lists runtime first.
    ordered = [name for name in selected if name not in ("runtime", "ffi")]
    if any(name != "ffi" for name in selected):
        ordered.append("runtime")
    libraries = _libraries(library_dir, platform, [_COMPONENTS[name] for name in ordered])
    ffi = _ffi_info()
    ffi_library = (
        ffi.find_windows_implib() if platform in ("Windows", "win32") else ffi.find_libtvm_ffi()
    )
    return [
        *(Path(lib["implib"] or lib["location"]) for lib in libraries.values()),
        Path(ffi_library),
    ]
