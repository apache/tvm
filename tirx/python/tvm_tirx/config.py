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
"""Discover TIRx-owned headers and libraries without loading native code.

The same file is installed beside the CMake package config.  CMake invokes it
there with an exact library directory, independently of Python package search.
"""

import argparse
import json
import sys
from pathlib import Path

_LIBRARY_NAMES = (
    "tvm_runtime",
    "tvm_tirx",
    "tvm_runtime_static",
    "tvm_runtime_cuda",
    "tvm_runtime_opencl",
    "tvm_runtime_vulkan",
    "tvm_runtime_metal",
    "tvm_runtime_rocm",
)


def _find_prefix(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "include/tvm/runtime/base.h").is_file():
            return candidate
    raise RuntimeError(f"Cannot find TIRx public headers above {start}")


def _paths(prefix: Path | None, library_dir: Path | None) -> tuple[Path, Path]:
    if library_dir is not None:
        library_dir = library_dir.resolve()
        prefix = prefix.resolve() if prefix is not None else _find_prefix(library_dir)
    else:
        prefix = prefix.resolve() if prefix is not None else _find_prefix(Path(__file__).parent)
        library_dir = prefix / "lib"
        for candidate in (prefix / "lib", prefix / "lib64", prefix / "build/lib"):
            if (candidate / "cmake/tvm_tirx/tvm_tirx-build.json").is_file() and any(
                (candidate / name).is_file()
                for name in ("libtvm_runtime.so", "libtvm_runtime.dylib", "tvm_runtime.dll")
            ):
                library_dir = candidate
                break
    if not (prefix / "include/tvm/runtime/base.h").is_file():
        raise RuntimeError(f"Cannot find TIRx public headers in {prefix / 'include'}")
    return prefix, library_dir


def _libraries(library_dir: Path, platform: str) -> dict[str, dict[str, str]]:
    libraries = {}
    windows = platform in ("Windows", "win32")
    apple = platform in ("Darwin", "iOS", "darwin")
    for name in _LIBRARY_NAMES:
        if name.endswith("_static"):
            filenames = (f"{name}.lib", f"lib{name}.a") if windows else (f"lib{name}.a",)
        elif windows:
            filenames = (f"{name}.dll",)
        elif apple:
            filenames = (f"lib{name}.dylib", f"lib{name}.so")
        else:
            filenames = (f"lib{name}.so",)
        location = next((library_dir / f for f in filenames if (library_dir / f).is_file()), None)
        if location is None:
            continue
        implib = ""
        if windows and not name.endswith("_static"):
            import_names = (f"{name}.lib", f"lib{name}.dll.a")
            import_library = next(
                (library_dir / f for f in import_names if (library_dir / f).is_file()), None
            )
            if import_library is None:
                if name in ("tvm_runtime", "tvm_tirx"):
                    raise RuntimeError(f"Missing import library for {location}")
                # Optional wheel sidecars may only support runtime loading.
                # Without an import library they cannot provide a native link target.
                continue
            implib = str(import_library)
        libraries[name] = {"location": str(location), "implib": implib}
    return libraries


def configuration(
    prefix: Path | None = None,
    library_dir: Path | None = None,
    platform: str = sys.platform,
) -> dict:
    """Return paths and consumer metadata for one installed TIRx prefix."""
    prefix, library_dir = _paths(prefix, library_dir)
    cmake_dir = library_dir / "cmake/tvm_tirx"
    metadata_path = cmake_dir / "tvm_tirx-build.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot read TIRx build metadata at {metadata_path}: {error}"
        ) from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("version"), str):
        raise RuntimeError(f"Invalid TIRx version metadata in {metadata_path}")
    if metadata.get("openmp") not in ("OFF", "gnu", "intel"):
        raise RuntimeError(f"Invalid TIRx OpenMP metadata in {metadata_path}")
    custom_logging = metadata.get("custom_logging", False)
    use_threads = metadata.get("use_threads", True)
    if not isinstance(custom_logging, bool) or not isinstance(use_threads, bool):
        raise RuntimeError(f"Invalid TIRx runtime interface metadata in {metadata_path}")
    libraries = _libraries(library_dir, platform)
    if "tvm_runtime" not in libraries:
        raise RuntimeError(f"Cannot find the TIRx runtime library in {library_dir}")
    return {
        "version": metadata["version"],
        "openmp": metadata["openmp"],
        "custom_logging": custom_logging,
        "use_threads": use_threads,
        "include_dir": str(prefix / "include"),
        "library_dir": str(library_dir),
        "cmake_dir": str(cmake_dir),
        "libraries": libraries,
    }


def __main__() -> None:
    """Print installed include, library, or CMake package paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, help="Use this exact installation prefix")
    parser.add_argument("--library-dir", type=Path, help="Use this exact native library directory")
    parser.add_argument("--platform", default=sys.platform, help=argparse.SUPPRESS)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument(
        "--includedir", action="store_true", help="Print the public include directory"
    )
    outputs.add_argument("--libdir", action="store_true", help="Print the native library directory")
    outputs.add_argument(
        "--libfiles", action="store_true", help="Print library files, one per line"
    )
    outputs.add_argument(
        "--cmakedir", action="store_true", help="Print the CMake package directory"
    )
    outputs.add_argument(
        "--json", action="store_true", help="Print all consumer configuration as JSON"
    )
    args = parser.parse_args()
    if not any((args.includedir, args.libdir, args.libfiles, args.cmakedir, args.json)):
        parser.print_help()
        return
    try:
        if args.json or args.libfiles:
            result = configuration(args.prefix, args.library_dir, args.platform)
            if args.json:
                print(json.dumps(result))
            else:
                for library in result["libraries"].values():
                    print(library["implib"] or library["location"])
        else:
            prefix, library_dir = _paths(args.prefix, args.library_dir)
            if args.includedir:
                print(prefix / "include")
            elif args.libdir:
                print(library_dir)
            else:
                print(library_dir / "cmake/tvm_tirx")
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    __main__()
