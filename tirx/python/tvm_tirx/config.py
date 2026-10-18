#!/usr/bin/env python3
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
import importlib.util
import json
import shlex
import sys
from pathlib import Path

if __package__:
    from . import libinfo
else:
    # The native CMake package installs the pure Python helper beside this file.
    spec = importlib.util.spec_from_file_location(
        "_tvm_tirx_libinfo", Path(__file__).with_name("libinfo.py")
    )
    libinfo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(libinfo)


def configuration(
    prefix: Path | None = None,
    library_dir: Path | None = None,
    platform: str = sys.platform,
) -> dict:
    """Return the public include and core shared-library paths for one installation."""
    return libinfo.configuration(prefix, library_dir, platform)


def __main__() -> None:
    """Print LLVM-style component configuration without loading native libraries."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Components default to all available components. Examples: --libs runtime; "
        "--libs runtime cuda; --libs compiler. Shared linking is the default.",
    )
    parser.add_argument("components", nargs="*", help="Components to link (default: all)")
    parser.add_argument("--install-prefix", type=Path, help="Query this exact installation prefix")
    parser.add_argument(
        "--library-dir", type=Path, help="Query this exact native library directory"
    )
    parser.add_argument("--platform", default=sys.platform, help=argparse.SUPPRESS)
    outputs = (
        "version",
        "prefix",
        "includedir",
        "libdir",
        "cmakedir",
        "components",
        "cflags",
        "cxxflags",
        "ldflags",
        "libs",
        "libnames",
        "libfiles",
        "json",
    )
    for output in outputs:
        parser.add_argument("--" + output, dest="show_" + output, action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--link-shared", action="store_true", help="Use shared libraries (default)")
    mode.add_argument(
        "--link-static", action="store_true", help="Require a complete static closure"
    )
    args = parser.parse_args()
    if not any(getattr(args, "show_" + output) for output in outputs):
        parser.print_help()
        return
    try:
        if args.link_static:
            raise RuntimeError(
                "Static linking is unavailable: this package does not provide a "
                "complete static dependency closure (including TVM FFI and its dependencies)"
            )
        if args.show_version:
            print(libinfo.__version__)
        if all(not getattr(args, "show_" + output) for output in outputs if output != "version"):
            return
        prefix, library_dir = libinfo._paths(args.install_prefix, args.library_dir)
        paths = {
            "prefix": prefix,
            "includedir": prefix / "include",
            "libdir": library_dir,
            "cmakedir": library_dir / "cmake/tvm_tirx",
        }
        if not paths["cmakedir"].is_dir() and (prefix / "cmake/tvm_tirx-config.cmake").is_file():
            paths["cmakedir"] = prefix / "cmake"
        for output, path in paths.items():
            if getattr(args, "show_" + output):
                print(path)
        if args.show_components:
            print(" ".join([*libinfo._available(library_dir, args.platform), "ffi"]))
        if args.show_json:
            print(json.dumps(configuration(prefix, library_dir, args.platform)))
        if args.show_cflags or args.show_cxxflags:
            ffi = libinfo._ffi_info()
            includes = dict.fromkeys(
                [str(prefix / "include"), ffi.find_include_path(), ffi.find_dlpack_include_path()]
            )
            flags = [*("-I" + path for path in includes), "-DTVM_INDEX_DEFAULT_I64=1"]
            if args.show_cflags:
                print(shlex.join(flags))
            if args.show_cxxflags:
                print(shlex.join([*flags, "-std=c++20"]))
        if any((args.show_ldflags, args.show_libs, args.show_libnames, args.show_libfiles)):
            libraries = libinfo._link_inputs(library_dir, args.platform, args.components)
            apple = args.platform in ("Darwin", "iOS", "darwin")
            if args.show_ldflags:
                print(
                    shlex.join(
                        [
                            "-L" + str(path)
                            for path in dict.fromkeys(lib.parent for lib in libraries)
                        ]
                    )
                )
            if args.show_libnames:
                print(shlex.join([lib.name for lib in libraries]))
            if args.show_libfiles:
                print(shlex.join([str(lib) for lib in libraries]))
            if args.show_libs:
                if apple:
                    flags = [
                        flag
                        for lib in libraries
                        for flag in ("-Xlinker", "-needed_library", "-Xlinker", str(lib))
                    ]
                elif args.platform.startswith("linux") or args.platform == "Linux":
                    flags = [
                        "-Wl,--push-state,--no-as-needed",
                        *map(str, libraries),
                        "-Wl,--pop-state",
                    ]
                else:
                    raise RuntimeError(f"--libs retention flags are unsupported on {args.platform}")
                print(shlex.join(flags))
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    __main__()
