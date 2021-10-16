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
# pylint: disable=invalid-name
"""Utility to invoke Xcode compiler toolchain"""
from __future__ import absolute_import as _abs

import os
import sys
import subprocess
import json
from .._ffi.base import py_str
from . import utils


def xcrun(cmd):
    """Run xcrun and return the output.

    Parameters
    ----------
    cmd : list of str
        The command sequence.

    Returns
    -------
    out : str
        The output string.
    """
    cmd = ["xcrun"] + cmd
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    return out.strip()


def __get_min_os_version(sdk):
    if sdk in ("macosx", "iphonesimulator"):
        return None
    if sdk == "iphoneos":
        return "13.0"
    raise RuntimeError("Unsupported sdk: %s" % sdk)


def __get_min_os_version_cmd(sdk, min_os_version):
    if min_os_version is None:
        min_os_version = __get_min_os_version(sdk)
    if min_os_version is not None:
        return "-mios-version-min=" + min_os_version
    return ""


def create_dylib(output, objects, arch, sdk="macosx", min_os_version=None):
    """Create dynamic library.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : list
        List of object files.

    options : str
        The additional options.

    arch : str
        Target major architectures

    sdk : str
        The sdk to be used.
    """
    clang = xcrun(["-sdk", sdk, "-find", "clang"])
    sdk_path = xcrun(["-sdk", sdk, "--show-sdk-path"])
    cmd = [clang]
    cmd += ["-dynamiclib"]
    cmd += ["-arch", arch]
    cmd += ["-isysroot", sdk_path]
    cmd += [__get_min_os_version_cmd(sdk, min_os_version)]
    cmd += ["-o", output]
    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


# assign so as default output format
create_dylib.output_format = "dylib"


def compile_metal(code, path_target=None, sdk="macosx", min_os_version=None):
    """Compile metal with CLI tool from env.

    Parameters
    ----------
    code : str
        The cuda code.

    path_target : str, optional
        Output file.

    sdk : str, optional
        The target platform SDK.

    Return
    ------
    metallib : bytearray
        The bytearray of the metallib
    """
    temp = utils.tempdir()
    temp_code = temp.relpath("my_lib.metal")
    temp_ir = temp.relpath("my_lib.air")
    temp_target = temp.relpath("my_lib.metallib")

    with open(temp_code, "w") as out_file:
        out_file.write(code)
    file_target = path_target if path_target else temp_target

    # See:
    # - https://developer.apple.com/documentation/metal/gpu_functions_libraries/building_a_library_with_metal_s_command-line_tools#overview # pylint: disable=line-too-long
    #
    #   xcrun -sdk macosx metal -c MyLibrary.metal -o MyLibrary.air
    #   xcrun -sdk macosx metallib MyLibrary.air -o MyLibrary.metallib
    min_target = __get_min_os_version_cmd(sdk, min_os_version)
    if sdk == "macosx":
        language_version = "-std=macos-metal2.3"
    elif sdk in ("iphoneos", "iphonesimulator"):
        language_version = "-std=ios-metal2.3"
    else:
        raise RuntimeError("Unsupported sdk: %s" % sdk)
    cmd1 = ["xcrun", "-sdk", sdk, "metal", language_version, min_target, "-O3"]
    cmd1 += ["-c", temp_code, "-o", temp_ir]
    cmd2 = ["xcrun", "-sdk", sdk, "metallib"]
    cmd2 += [temp_ir, "-o", file_target]
    proc = subprocess.Popen(
        " ".join(cmd1) + ";" + " ".join(cmd2),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Compilation error:\n")
        sys.stderr.write(py_str(out))
        sys.stderr.flush()
        libbin = None
    else:
        libbin = bytearray(open(file_target, "rb").read())
    return libbin


def compile_coreml(model, model_name="main", out_dir="."):
    """Compile coreml model and return the compiled model path."""
    mlmodel_path = os.path.join(out_dir, model_name + ".mlmodel")
    mlmodelc_path = os.path.join(out_dir, model_name + ".mlmodelc")
    metadata = {"inputs": list(model.input_description), "outputs": list(model.output_description)}
    # Use the description field to send info to CoreML runtime
    model.short_description = json.dumps(metadata)
    model.save(mlmodel_path)

    res = xcrun(["coremlcompiler", "compile", mlmodel_path, out_dir])
    if not os.path.isdir(mlmodelc_path):
        raise RuntimeError("Compile failed: %s" % res)

    return mlmodelc_path
