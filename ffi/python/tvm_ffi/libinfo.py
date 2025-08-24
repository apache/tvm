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

import sys
import os
import glob


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
    ffi_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    dll_path = [os.path.join(ffi_dir, "lib")]
    dll_path += [os.path.join(ffi_dir, "..", "..", "build", "lib")]
    # in source build from parent if needed
    dll_path += [os.path.join(ffi_dir, "..", "..", "..", "build", "lib")]

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(split_env_var("LD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(split_env_var("PATH", ";"))
    return [os.path.abspath(x) for x in dll_path if os.path.isdir(x)]


def find_libtvm_ffi():
    """Find libtvm_ffi."""
    dll_path = get_dll_directories()
    if sys.platform.startswith("win32"):
        lib_dll_names = ["tvm_ffi.dll"]
    elif sys.platform.startswith("darwin"):
        lib_dll_names = ["libtvm_ffi.dylib", "libtvm_ffi.so"]
    else:
        lib_dll_names = ["libtvm_ffi.so"]

    name = lib_dll_names
    lib_dll_path = [os.path.join(p, name) for name in lib_dll_names for p in dll_path]
    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_found:
        raise RuntimeError(f"Cannot find library: {name}\nList of candidates:\n{lib_dll_path}")

    return lib_found[0]


def find_source_path():
    """Find packaged source home path."""
    candidates = [
        os.path.join(os.path.dirname(os.path.realpath(__file__))),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."),
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "cmake")):
            return candidate
    raise RuntimeError("Cannot find home path.")


def find_cmake_path():
    """Find the preferred cmake path."""
    candidates = [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib", "cmake"),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "cmake"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError("Cannot find cmake path.")


def find_include_path():
    """Find header files for C compilation."""
    candidates = [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "include"),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "include"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError("Cannot find include path.")


def find_dlpack_include_path():
    """Find dlpack header files for C compilation."""
    install_include_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "include")
    if os.path.isdir(os.path.join(install_include_path, "dlpack")):
        return install_include_path

    source_include_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "3rdparty", "dlpack", "include"
    )
    if os.path.isdir(source_include_path):
        return source_include_path

    raise RuntimeError("Cannot find include path.")


def find_cython_lib():
    """Find the path to tvm cython."""
    path_candidates = [
        os.path.dirname(os.path.realpath(__file__)),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "build"),
    ]
    suffixes = "pyd" if sys.platform.startswith("win32") else "so"
    for candidate in path_candidates:
        for path in glob.glob(os.path.join(candidate, f"core*.{suffixes}")):
            return os.path.abspath(path)
    raise RuntimeError("Cannot find tvm cython path.")
