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
import subprocess
from tvm._ffi.base import py_str
from tvm._ffi.libinfo import find_lib_path


def create_tvmjs_wasm(output, objects, options=None, cc="emcc"):
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
    """
    cmd = [cc]
    cmd += ["-O3"]

    cmd += ["-std=c++14"]
    cmd += ["--no-entry"]
    cmd += ["-s", "ERROR_ON_UNDEFINED_SYMBOLS=0"]
    cmd += ["-s", "STANDALONE_WASM=1"]
    cmd += ["-s", "ALLOW_MEMORY_GROWTH=1"]

    objects = [objects] if isinstance(objects, str) else objects

    with_runtime = False
    for obj in objects:
        if obj.find("wasm_runtime.bc") != -1:
            with_runtime = True

    if not with_runtime:
        objects += [find_lib_path("wasm_runtime.bc")[0]]

    objects += [find_lib_path("tvmjs_support.bc")[0]]
    objects += [find_lib_path("webgpu_runtime.bc")[0]]

    cmd += ["-o", output]
    cmd += objects

    if options:
        cmd += options

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


create_tvmjs_wasm.object_format = "bc"
