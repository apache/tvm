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
"""Config utilities for finding paths to lib and headers"""

import argparse
import sys
import os
from . import libinfo


def find_windows_implib():
    libdir = os.path.dirname(libinfo.find_libtvm_ffi())
    implib = os.path.join(libdir, "tvm_ffi.lib")
    if not os.path.isfile(implib):
        raise RuntimeError(f"Cannot find imp lib {implib}")
    return implib


def __main__():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Get various configuration information needed to compile with tvm-ffi",
    )

    parser.add_argument("--includedir", action="store_true", help="Print include directory")
    parser.add_argument(
        "--dlpack-includedir", action="store_true", help="Print dlpack include directory"
    )
    parser.add_argument("--cmakedir", action="store_true", help="Print library directory")
    parser.add_argument("--sourcedir", action="store_true", help="Print source directory")
    parser.add_argument("--libfiles", action="store_true", help="Fully qualified library filenames")
    parser.add_argument("--libdir", action="store_true", help="Print library directory")
    parser.add_argument("--libs", action="store_true", help="Libraries to be linked")
    parser.add_argument("--cython-lib-path", action="store_true", help="Print cython path")
    parser.add_argument("--cxxflags", action="store_true", help="Print cxx flags")
    parser.add_argument("--ldflags", action="store_true", help="Print ld flags")

    args = parser.parse_args()

    # print help when no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.includedir:
        print(libinfo.find_include_path())
    if args.dlpack_includedir:
        print(libinfo.find_dlpack_include_path())
    if args.cmakedir:
        print(libinfo.find_cmake_path())
    if args.libdir:
        print(os.path.dirname(libinfo.find_libtvm_ffi()))
    if args.libfiles:
        if sys.platform.startswith("win32"):
            print(find_windows_implib())
        else:
            print(libinfo.find_libtvm_ffi())
    if args.sourcedir:
        print(libinfo.find_source_path())
    if args.cython_lib_path:
        print(libinfo.find_cython_lib())
    if args.cxxflags:
        include_dir = libinfo.find_include_path()
        dlpack_include_dir = libinfo.find_dlpack_include_path()
        print(f"-I{include_dir} -I{dlpack_include_dir} -std=c++17")
    if args.libs:
        if sys.platform.startswith("win32"):
            print(find_windows_implib())
        else:
            print("-ltvm_ffi")

    if args.ldflags:
        if not sys.platform.startswith("win32"):
            print(f"-L{os.path.dirname(libinfo.find_libtvm_ffi())}")


if __name__ == "__main__":
    __main__()
