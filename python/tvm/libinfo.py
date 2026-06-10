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
"""Library information."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tvm_ffi import libinfo as tvm_ffi_libinfo
from tvm_ffi.libinfo import load_lib_ctypes


def use_runtime_lib() -> bool:
    """Whether ``TVM_USE_RUNTIME_LIB`` requests runtime-only mode.

    Recognises ``1`` / ``true`` / ``yes`` (case-insensitive) as truthy.
    Anything else — including ``0`` and the unset case — is False.
    """
    return os.environ.get("TVM_USE_RUNTIME_LIB", "0").lower() in ("1", "true", "yes")


def _rel_top_directory() -> Path:
    """Get the current directory of this file."""
    return Path(__file__).parent


def _dev_top_directory() -> Path:
    """Get the top-level development directory."""
    return _rel_top_directory() / ".." / ".."


def package_lib_paths() -> list[Path]:
    """Return search directories for TVM's shared libraries.

    Anchored on this file's location (``python/tvm/libinfo.py``), the list
    covers the wheel-install layout (``python/tvm/lib/``) and the in-tree dev
    build layouts (``<worktree>/build/lib/`` and ``<worktree>/lib/``).
    ``TVM_LIBRARY_PATH`` is prepended when set so it takes priority. Callers
    pick the basenames they want (e.g. ``libtvm_runtime.so``) and the load
    mode; this function only returns the search path.
    """
    paths: list[Path] = []
    if os.environ.get("TVM_LIBRARY_PATH"):
        paths.append(Path(os.environ["TVM_LIBRARY_PATH"]))
    paths += [
        _rel_top_directory() / "lib",  # wheel layout
        _dev_top_directory() / "build" / "lib",  # dev: <worktree>/build/lib
        _dev_top_directory() / "lib",  # dev: <worktree>/lib
    ]
    return paths


_BACKEND_RUNTIME_LIBS = ["cuda", "vulkan", "opencl", "metal", "rocm", "hexagon", "extra"]


def load_backend_libs(runtime_lib_path: str) -> None:
    """Try to load each known backend runtime DSO; failures are silent."""
    runtime_dir = Path(runtime_lib_path).resolve().parent
    for backend in _BACKEND_RUNTIME_LIBS:
        try:
            load_lib_ctypes(
                package="tvm",
                target_name=f"tvm_runtime_{backend}",
                mode="RTLD_GLOBAL",
                extra_lib_paths=[runtime_dir],
            )
        except (OSError, FileNotFoundError, RuntimeError):
            pass


def split_env_var(env_var, split):
    """Splits environment variable string.

    Parameters
    ----------
    env_var : str
        Name of environment variable.

    split : str
        String to split env_var on.

    Returns
    -------
    splits : list(string)
        If env_var exists, split env_var. Otherwise, empty list.
    """
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(split)]
    return []


def _lib_search_directories() -> list[Path]:
    """Return directories searched when locating libraries and web assets by name."""
    paths = package_lib_paths()
    for top in (_rel_top_directory(), _dev_top_directory()):
        paths += [
            top / "build" / "lib" / "Release",  # Windows CMake build
            top / "build" / "Release",
            top / "build",
            top / "build" / "3rdparty" / "cutlass_fpA_intB_gemm" / "cutlass_kernels",
            top / "build" / "3rdparty" / "libflash_attn" / "src",
            top / "web" / "dist" / "wasm",
            top / "web" / "dist",
        ]
    if sys.platform.startswith("win32"):
        paths += [Path(p) for p in split_env_var("PATH", ";")]
    elif sys.platform.startswith("darwin"):
        paths += [Path(p) for p in split_env_var("DYLD_LIBRARY_PATH", ":")]
        paths += [Path(p) for p in split_env_var("PATH", ":")]
    else:
        paths += [Path(p) for p in split_env_var("LD_LIBRARY_PATH", ":")]
        paths += [Path(p) for p in split_env_var("PATH", ":")]
    return paths


_TVM_LIB_TARGETS = ["tvm_compiler", "tvm_runtime"]
_TVM_EXT_LIB_TARGETS = ["fpA_intB_gemm", "flash_attn"]


def find_lib_path(name=None, optional=False):
    """Find dynamic library files.

    Parameters
    ----------
    name : str or list of str, optional
        File name(s) to search for across the TVM library directories. When
        None, search for TVM's own compiler/runtime libraries (plus the
        optional extension libraries) by their platform-specific names.
    optional : bool
        If True, return None instead of raising when nothing is found.

    Returns
    -------
    lib_path : list(string)
        List of all found paths to the libraries.
    """
    search_dirs = _lib_search_directories()

    if name is not None:
        # Honour explicit names directly regardless of TVM_USE_RUNTIME_LIB;
        # that env var is interpreted by ``base.py::_load_lib`` to choose
        # which name to ask for.
        names = name if isinstance(name, list) else [name]
        lib_found = []
        for search_dir in search_dirs:
            for n in names:
                try:
                    candidate = (search_dir / n).resolve()
                    found = candidate.is_file()
                except OSError:
                    continue
                if found and str(candidate) not in lib_found:
                    lib_found.append(str(candidate))
    else:
        use_runtime = use_runtime_lib()
        names = ["tvm_runtime"] if use_runtime else _TVM_LIB_TARGETS + _TVM_EXT_LIB_TARGETS
        lib_found = []
        for target in names:
            try:
                lib_found.append(
                    str(
                        tvm_ffi_libinfo._find_library_by_basename(
                            "tvm", target, extra_lib_paths=search_dirs
                        )
                    )
                )
            except RuntimeError:
                continue
        if use_runtime and lib_found:
            sys.stderr.write(f"Loading runtime library {lib_found[0]}... exec only\n")
            sys.stderr.flush()

    if not lib_found:
        if optional:
            return None
        raise RuntimeError(
            f"Cannot find libraries: {names}\n"
            + "List of directories searched:\n"
            + "\n".join(str(p) for p in search_dirs)
        )
    return lib_found


def find_tvm_include_path() -> str:
    """Find TVM's own public header directory for C compilation."""
    if ret := tvm_ffi_libinfo._resolve_and_validate(
        paths=[
            _rel_top_directory() / "include",  # Standard install
            _dev_top_directory() / "include",  # Development mode
        ],
        cond=lambda p: (p / "tvm" / "runtime").is_dir(),
    ):
        return ret
    raise RuntimeError("Cannot find TVM include path.")


def find_include_path(optional=False):
    """Find header files for C compilation.

    Parameters
    ----------
    optional : bool
        If True, return None instead of raising when headers cannot be found.

    Returns
    -------
    include_path : list(string)
        List of all found paths to header files.
    """
    try:
        return [find_tvm_include_path(), *tvm_ffi_libinfo.include_paths()]
    except RuntimeError:
        if optional:
            return None
        raise


# The version is written by setuptools_scm into _version.py at build time
# (see [tool.setuptools_scm] in pyproject.toml). The fallback keeps a source
# checkout with no build run importable.
try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - source tree without a build
    __version__ = "0.25.dev0"
