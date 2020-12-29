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
"""Utility for Interacting with SPIRV Tools"""
import subprocess
import os
from . import utils
from .._ffi.base import py_str


def optimize(spv_bin):
    """Optimize SPIRV using spirv-opt via CLI

    Note that the spirv-opt is still experimental.

    Parameters
    ----------
    spv_bin : bytearray
        The spirv file

    Return
    ------
    cobj_bin : bytearray
        The HSA Code Object
    """

    tmp_dir = utils.tempdir()
    tmp_in = tmp_dir.relpath("input.spv")
    tmp_out = tmp_dir.relpath("output.spv")
    with open(tmp_in, "wb") as out_file:
        out_file.write(bytes(spv_bin))

    sdk = os.environ.get("VULKAN_SDK", None)
    cmd = os.path.join(sdk, "bin/spirv-opt") if sdk else "spirv-opt"
    args = [cmd, "-O", tmp_in, "-o", tmp_out]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Opitmizationerror using spirv-opt:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    return bytearray(open(tmp_out, "rb").read())
