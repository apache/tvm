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
"""Util to invoke emscripten compilers in the system."""
# pylint: disable=invalid-name
import os
import subprocess
from pathlib import Path

from tvm._ffi.base import py_str
from tvm._ffi.libinfo import find_lib_path


def create_tvmjs_wasm(output, objects, options=None, cc="emcc", libs=None):
    """Create wasm that is supposed to run with the tvmjs.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : list
        List of object files.

    options : str
        The additional options.

    cc : str, optional
        The compile string.

    libs : list
        List of user-defined library files (e.g. .bc files) to add into the wasm.
    """
    cmd = [cc]
    cmd += ["-O3"]
    cmd += ["-std=c++17"]
    cmd += ["--no-entry"]
    # NOTE: asynctify conflicts with wasm-exception
    # so we temp disable exception handling for now
    #
    # We also expect user to explicitly pass in
    # -s ASYNCIFY=1 as it can increase wasm size by 2xq
    #
    # cmd += ["-s", "ASYNCIFY=1"]
    # cmd += ["-fwasm-exceptions"]
    cmd += ["-s", "WASM_BIGINT=1"]
    cmd += ["-s", "ERROR_ON_UNDEFINED_SYMBOLS=0"]
    cmd += ["-s", "STANDALONE_WASM=1"]
    cmd += ["-s", "ALLOW_MEMORY_GROWTH=1"]
    cmd += ["-s", "TOTAL_MEMORY=160MB"]

    objects = [objects] if isinstance(objects, str) else objects

    with_runtime = False
    for obj in objects:
        if obj.find("wasm_runtime.bc") != -1:
            with_runtime = True

    all_libs = []
    if not with_runtime:
        all_libs += [find_lib_path("wasm_runtime.bc")[0]]

    all_libs += [find_lib_path("tvmjs_support.bc")[0]]
    all_libs += [find_lib_path("webgpu_runtime.bc")[0]]

    if libs:
        if not isinstance(libs, list):
            raise ValueError("Expect `libs` to be a list of paths in string.")
        for lib in libs:
            if not Path(lib).exists():
                raise RuntimeError(
                    "Cannot find file from libs:" + lib + "\n Try pass in an absolute path."
                )
        all_libs += libs

    cmd += ["-o", output]

    # let libraries go before normal object
    cmd += all_libs + objects

    if options:
        cmd += options

    is_windows = os.name == "nt"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=is_windows)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


create_tvmjs_wasm.object_format = "bc"
