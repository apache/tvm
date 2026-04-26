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

import ctypes
import importlib.metadata as im
import os
import sys
from pathlib import Path


def use_runtime_lib() -> bool:
    """Whether ``TVM_USE_RUNTIME_LIB`` requests runtime-only mode.

    Recognises ``1`` / ``true`` / ``yes`` (case-insensitive) as truthy.
    Anything else — including ``0`` and the unset case — is False.
    """
    return os.environ.get("TVM_USE_RUNTIME_LIB", "0").lower() in ("1", "true", "yes")


def package_lib_paths() -> list[Path]:
    """Return search directories for TVM's shared libraries.

    Anchored on this file's location (``python/tvm/libinfo.py``), the list
    covers the wheel-install layout (``python/tvm/lib/``) and the in-tree dev
    build layouts (``<worktree>/build/lib/`` and ``<worktree>/lib/``). Callers
    pick the basenames they want (e.g. ``libtvm_runtime.so``) and the load
    mode; this function only returns the search path.
    """
    pkg = Path(__file__).parent  # python/tvm/
    return [
        pkg / "lib",  # wheel layout
        pkg.parent.parent / "build" / "lib",  # dev: <worktree>/build/lib
        pkg.parent.parent / "lib",  # dev: <worktree>/lib
    ]


# Mirror of ``tvm_ffi.libinfo.{load_lib_ctypes,_find_library_by_basename}`` with
# the ``extra_lib_paths`` parameter from apache/tvm-ffi#570 so dev-mode lookups
# anchor on the *caller's* package root rather than tvm-ffi's own ``__file__``.
# Once apache/tvm-ffi#570 lands and the submodule bumps, drop these and switch
# ``base.py`` back to ``from tvm_ffi.libinfo import load_lib_ctypes``.


def _find_library_by_basename(
    package: str,
    target_name: str,
    extra_lib_paths: list[Path] | None = None,
) -> Path:
    """Resolve ``lib<target_name>.{so,dylib,dll}`` for ``package``.

    Search order: wheel-install RECORD walk → caller-supplied
    ``extra_lib_paths`` → ``PATH`` / ``LD_LIBRARY_PATH`` /
    ``DYLD_LIBRARY_PATH``. Raises ``RuntimeError`` listing every candidate
    directory tried if nothing matches.
    """
    if sys.platform.startswith("win32"):
        lib_dll_names = (f"{target_name}.dll",)
    elif sys.platform.startswith("darwin"):
        lib_dll_names = (f"lib{target_name}.dylib", f"lib{target_name}.so")
    else:
        lib_dll_names = (f"lib{target_name}.so",)

    try:
        dist = im.distribution(package)
        record = dist.read_text("RECORD") or ""
        for line in record.splitlines():
            partial_path, *_ = line.split(",")
            if partial_path.endswith(lib_dll_names):
                try:
                    path = (dist._path.parent / partial_path).resolve()
                except OSError:
                    continue
                if path.name in lib_dll_names and path.is_file():
                    return path
    except (im.PackageNotFoundError, OSError):
        pass

    dll_paths: list[Path] = []
    if extra_lib_paths is not None:
        for i, p in enumerate(extra_lib_paths):
            if not isinstance(p, Path):
                raise TypeError(
                    f"extra_lib_paths[{i}] must be a pathlib.Path, got {type(p).__name__}: {p!r}"
                )
        dll_paths.extend(extra_lib_paths)

    if sys.platform.startswith("win32"):
        dll_paths.extend(Path(p) for p in split_env_var("PATH", ";"))
    elif sys.platform.startswith("darwin"):
        dll_paths.extend(Path(p) for p in split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_paths.extend(Path(p) for p in split_env_var("PATH", ":"))
    else:
        dll_paths.extend(Path(p) for p in split_env_var("LD_LIBRARY_PATH", ":"))
        dll_paths.extend(Path(p) for p in split_env_var("PATH", ":"))

    for d in dll_paths:
        for name in lib_dll_names:
            try:
                path = (d / name).resolve()
            except OSError:
                continue
            if path.is_file():
                return path

    raise RuntimeError(
        f"Cannot find library {', '.join(lib_dll_names)}; searched directories:\n  "
        + "\n  ".join(str(p) for p in dll_paths)
    )


def load_lib_ctypes(
    package: str,
    target_name: str,
    mode: str,
    extra_lib_paths: list[Path] | None = None,
) -> ctypes.CDLL:
    """Locate and ``ctypes.CDLL``-load ``lib<target_name>`` for ``package``.

    ``mode`` is one of ``"RTLD_LOCAL"`` / ``"RTLD_GLOBAL"`` (resolved against
    ``ctypes``). On Windows, the library's directory is registered via
    ``os.add_dll_directory`` before the load.
    """
    lib_path = _find_library_by_basename(package, target_name, extra_lib_paths)
    if sys.platform.startswith("win32"):
        os.add_dll_directory(str(lib_path.parent))
    return ctypes.CDLL(str(lib_path), getattr(ctypes, mode))


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


def get_dll_directories():
    """Get the possible dll directories"""
    # NB: This will either be the source directory (if TVM is run
    # inplace) or the install directory (if TVM is installed).
    # An installed TVM's curr_path will look something like:
    #   $PREFIX/lib/python3.6/site-packages/tvm/_ffi
    ffi_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..")
    install_lib_dir = os.path.join(ffi_dir, "..", "..", "..")

    dll_path = []

    if os.environ.get("TVM_LIBRARY_PATH", None):
        dll_path.append(os.environ["TVM_LIBRARY_PATH"])

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(split_env_var("LD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(split_env_var("PATH", ";"))

    # Pip lib directory
    dll_path.append(ffi_dir)
    dll_path.append(os.path.join(ffi_dir, "lib"))
    # Default CMake build directory: shared libs are placed under build/lib/
    # to mirror the tvm-ffi layout (so wheel install + dev-mode dlopen find
    # them via the same `lib/` subdir).
    dll_path.append(os.path.join(source_dir, "build", "lib"))
    dll_path.append(os.path.join(source_dir, "build", "lib", "Release"))
    dll_path.append(os.path.join(source_dir, "build"))
    dll_path.append(os.path.join(source_dir, "build", "Release"))
    # Default make build directory
    dll_path.append(os.path.join(source_dir, "lib"))

    dll_path.append(install_lib_dir)

    # use extra TVM_HOME environment for finding libraries.
    if os.environ.get("TVM_HOME", None):
        tvm_source_home_dir = os.environ["TVM_HOME"]
    else:
        tvm_source_home_dir = source_dir

    if os.path.isdir(tvm_source_home_dir):
        dll_path.append(os.path.join(tvm_source_home_dir, "web", "dist", "wasm"))
        dll_path.append(os.path.join(tvm_source_home_dir, "web", "dist"))

    dll_path = [os.path.realpath(x) for x in dll_path]
    return [x for x in dll_path if os.path.isdir(x)]


def find_lib_path(name=None, search_path=None, optional=False):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    use_runtime = use_runtime_lib()
    dll_path = get_dll_directories()
    # When the caller asks for a specific ``name`` we honour it directly
    # regardless of TVM_USE_RUNTIME_LIB; that env var is interpreted by
    # ``base.py::_load_lib`` to choose which name to ask for. This avoids
    # the runtime/compiler dual-list logic below from making `name` paths
    # unreachable when the user sets TVM_USE_RUNTIME_LIB.
    if name is not None:
        use_runtime = False

    if search_path is not None:
        if isinstance(search_path, list):
            dll_path = dll_path + search_path
        else:
            dll_path.append(search_path)

    if name is not None:
        if isinstance(name, list):
            lib_dll_path = []
            for n in name:
                lib_dll_path += [os.path.join(p, n) for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, name) for p in dll_path]
        runtime_dll_path = []
        ext_lib_dll_path = []
    else:
        if sys.platform.startswith("win32"):
            lib_dll_names = ["libtvm_compiler.dll", "tvm_compiler.dll"]
            runtime_dll_names = ["libtvm_runtime.dll", "tvm_runtime.dll"]
            ext_lib_dll_names = [
                "3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/libfpA_intB_gemm.dll",
                "3rdparty/libflash_attn/src/libflash_attn.dll",
            ]
        elif sys.platform.startswith("darwin"):
            lib_dll_names = ["libtvm_compiler.dylib"]
            runtime_dll_names = ["libtvm_runtime.dylib"]
            ext_lib_dll_names = []
        else:
            lib_dll_names = ["libtvm_compiler.so"]
            runtime_dll_names = ["libtvm_runtime.so"]
            ext_lib_dll_names = [
                "3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/libfpA_intB_gemm.so",
                "3rdparty/libflash_attn/src/libflash_attn.so",
            ]

        name = lib_dll_names + runtime_dll_names + ext_lib_dll_names
        lib_dll_path = [
            os.path.join(p, name)
            for name in lib_dll_names
            for p in dll_path
            if not p.endswith("python/tvm")
        ]
        runtime_dll_path = [
            os.path.join(p, name)
            for name in runtime_dll_names
            for p in dll_path
            if not p.endswith("python/tvm")
        ]
        ext_lib_dll_path = [os.path.join(p, name) for name in ext_lib_dll_names for p in dll_path]
    if not use_runtime:
        # try to find lib_dll_path
        lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]
        lib_found += [p for p in runtime_dll_path if os.path.exists(p) and os.path.isfile(p)]
        lib_found += [p for p in ext_lib_dll_path if os.path.exists(p) and os.path.isfile(p)]
    else:
        # try to find runtime_dll_path
        use_runtime = True
        lib_found = [p for p in runtime_dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_found:
        if not optional:
            message = (
                f"Cannot find libraries: {name}\n"
                + "List of candidates:\n"
                + "\n".join(lib_dll_path + runtime_dll_path)
            )
            raise RuntimeError(message)
        return None

    if use_runtime:
        sys.stderr.write(f"Loading runtime library {lib_found[0]}... exec only\n")
        sys.stderr.flush()
    return lib_found


def find_include_path(name=None, search_path=None, optional=False):
    """Find header files for C compilation.

    Parameters
    ----------
    name : list of str
        List of directory names to be searched.

    Returns
    -------
    include_path : list(string)
        List of all found paths to header files.
    """
    if os.environ.get("TVM_SOURCE_DIR", None):
        source_dir = os.environ["TVM_SOURCE_DIR"]
    elif os.environ.get("TVM_HOME", None):
        source_dir = os.environ["TVM_HOME"]
    else:
        ffi_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        for source_dir in ["..", "../..", "../../.."]:
            source_dir = os.path.join(ffi_dir, source_dir)
            if os.path.isdir(os.path.join(source_dir, "include")):
                break
        else:
            raise AssertionError(f"Cannot find the source directory given ffi_dir: {ffi_dir}")
    third_party_dir = os.path.join(source_dir, "3rdparty")

    header_path = []

    if os.environ.get("TVM_INCLUDE_PATH", None):
        header_path.append(os.environ["TVM_INCLUDE_PATH"])

    header_path.append(source_dir)
    header_path.append(third_party_dir)

    header_path = [os.path.abspath(x) for x in header_path]
    if search_path is not None:
        if isinstance(search_path, list):
            header_path = header_path + search_path
        else:
            header_path.append(search_path)
    if name is not None:
        if isinstance(name, list):
            tvm_include_path = []
            for n in name:
                tvm_include_path += [os.path.join(p, n) for p in header_path]
        else:
            tvm_include_path = [os.path.join(p, name) for p in header_path]
        dlpack_include_path = []
    else:
        tvm_include_path = [os.path.join(p, "include") for p in header_path]
        tvm_ffi_include_path = [
            os.path.join(p, "3rdparty", "tvm-ffi", "include") for p in header_path
        ]
        dlpack_include_path = [
            os.path.join(p, "3rdparty", "tvm-ffi", "3rdparty", "dlpack", "include")
            for p in header_path
        ]

        # try to find include path
        include_found = [p for p in tvm_include_path if os.path.exists(p) and os.path.isdir(p)]
        include_found += [p for p in tvm_ffi_include_path if os.path.exists(p) and os.path.isdir(p)]
        include_found += [p for p in dlpack_include_path if os.path.exists(p) and os.path.isdir(p)]

    if not include_found:
        message = (
            "Cannot find the files.\n"
            + "List of candidates:\n"
            + str("\n".join(tvm_include_path + dlpack_include_path))
        )
        if not optional:
            raise RuntimeError(message)
        return None

    return include_found


# current version
# We use the version of the incoming release for code
# that is under development.
# The following line is set by version.py
__version__ = "0.24.dev0"
