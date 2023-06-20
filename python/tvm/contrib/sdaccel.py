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
"""Utility for Interacting with SDAccel Tools"""
import os
import subprocess

import tvm._ffi

from . import utils


@tvm._ffi.register_func("tvm_callback_sdaccel_compile")
def compile_vhls(kernel_info, target):
    """Compile Vivado HLS code for SDAccel.

    Parameters
    ----------
    kernel_info : list of (str, str)
        List of kernel information.  The kernel information is a tuple of
        function name and source code.

    target : tvm.target.Target
        The compilation target

    Return
    ------
    xclbin : bytearray
        The bytearray of the xclbin
    """
    device_name = target.attrs.get("device", "")
    tmp_dir = utils.tempdir()

    sdk = os.environ.get("XILINX_SDX", None)
    xocc = os.path.join(sdk, "bin/xocc") if sdk else "xocc"
    target = os.environ.get(
        "XCL_TARGET", "sw_emu" if os.environ.get("XCL_EMULATION_MODE") else "hw"
    )
    advanced_params = [
        "--xp",
        "param:compiler.preserveHlsOutput=1",
        "--xp",
        "param:compiler.generateExtraRunData=true",
    ]
    platform = device_name
    if not platform:
        platform = os.environ.get("XCL_PLATFORM", os.environ.get("AWS_PLATFORM"))

    if platform is None:
        raise RuntimeError("No Xilinx device specified.")

    tmp_xo_files = []
    for funcname, code in kernel_info:
        funcname = funcname.value
        code = code.value

        tmp_cpp = tmp_dir.relpath(funcname + ".cpp")
        tmp_xo = tmp_dir.relpath(funcname + ".xo")

        with open(tmp_cpp, "wb") as out_file:
            out_file.write(bytes(code))

        # build xo
        args = (
            [xocc, "-c", "-t", target, "--platform", platform, "-o", tmp_xo, "-k", funcname]
            + advanced_params
            + [tmp_cpp]
        )
        returncode = subprocess.call(args)
        if returncode != 0:
            raise RuntimeError("Compile error")

        tmp_xo_files.append(tmp_xo)

    # build xclbin
    tmp_xclbin = tmp_dir.relpath("output.xclbin")
    args = (
        [xocc, "-l", "-t", target, "--platform", platform, "-o", tmp_xclbin]
        + tmp_xo_files
        + advanced_params
    )
    returncode = subprocess.call(args)
    if returncode != 0:
        raise RuntimeError("Link error")

    return bytearray(open(tmp_xclbin, "rb").read())
