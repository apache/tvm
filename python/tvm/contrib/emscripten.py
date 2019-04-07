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
from __future__ import absolute_import as _abs

import subprocess
from .._ffi.base import py_str
from .._ffi.libinfo import find_lib_path

def create_js(output,
              objects,
              options=None,
              side_module=False,
              cc="emcc"):
    """Create emscripten javascript library.

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
    cmd += ["-Oz"]
    if not side_module:
        cmd += ["-s", "RESERVED_FUNCTION_POINTERS=2"]
        cmd += ["-s", "NO_EXIT_RUNTIME=1"]
        extra_methods = ['cwrap', 'getValue', 'setValue', 'addFunction']
        cfg = "[" + (','.join("\'%s\'" % x for x in extra_methods)) + "]"
        cmd += ["-s", "EXTRA_EXPORTED_RUNTIME_METHODS=" + cfg]
    else:
        cmd += ["-s", "SIDE_MODULE=1"]
    cmd += ["-o", output]
    objects = [objects] if isinstance(objects, str) else objects
    with_runtime = False
    for obj in objects:
        if obj.find("libtvm_web_runtime.bc") != -1:
            with_runtime = True

    if not with_runtime and not side_module:
        objects += [find_lib_path("libtvm_web_runtime.bc")[0]]

    cmd += objects

    if options:
        cmd += options

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

create_js.object_format = "bc"
