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

"""Cross compilation for MicroTVM"""

from __future__ import absolute_import

import subprocess

from .._ffi.function import _init_api
from .._ffi.base import py_str


def create_lib(output, sources, options=None, compile_cmd="gcc"):
    """Compiles source code into a binary object file

    Parameters
    ----------
    output : str
        target library path

    sources : list
        list of source files to be compiled

    options: list
        list of additional option strings

    compile_cmd : str, optional
        compiler string
    """
    cmd = [compile_cmd]
    cmd += ["-c"]
    cmd += ["-g"]
    cmd += ["-o", output]
    if isinstance(sources, str):
        cmd += [sources]
    else:
        cmd += sources
    if options:
        cmd += options
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Error in compilation:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


_init_api("tvm.micro.cross_compile")
