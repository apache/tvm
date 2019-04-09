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
"""Util to invoke c++ compilers in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import sys
import subprocess
import os

from .._ffi.base import py_str
from .util import tempdir


def create_shared(output,
                  objects,
                  options=None,
                  cc="g++"):
    """Create shared library.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : list
        List of object files.

    options : list
        The list of additional options string.

    cc : str, optional
        The compile string.
    """
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        _linux_shared(output, objects, options, cc)
    elif sys.platform == "win32":
        _windows_shared(output, objects, options)
    else:
        raise ValueError("Unsupported platform")


# assign so as default output format
create_shared.output_format = "so" if sys.platform != "win32" else "dll"


def cross_compiler(cc, options=None, output_format="so"):
    """Create a cross compiler function.

    Parameters
    ----------
    cc :  str
        The cross compiler name.

    options : list, optional
        List of additional optional string.

    output_format : str, optional
        Library output format.

    Returns
    -------
    fcompile : function
        A compilation function that can be passed to export_library.
    """
    def _fcompile(outputs, objects, opts=None):
        opts = opts if opts else []
        if options:
            opts += options
        _linux_shared(outputs, objects, opts, cc=cc)
    _fcompile.output_format = output_format
    return _fcompile


def _linux_shared(output, objects, options, cc="g++"):
    cmd = [cc]
    cmd += ["-shared", "-fPIC"]
    if sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]
    cmd += ["-o", output]
    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects
    if options:
        cmd += options
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


def _windows_shared(output, objects, options):
    cl_cmd = ["cl"]
    cl_cmd += ["-c"]
    if isinstance(objects, str):
        objects = [objects]
    cl_cmd += objects
    if options:
        cl_cmd += options

    temp = tempdir()
    dllmain_path = temp.relpath("dllmain.cc")
    with open(dllmain_path, "w") as dllmain_obj:
        dllmain_obj.write('#include <windows.h>\
BOOL APIENTRY DllMain( HMODULE hModule,\
                       DWORD  ul_reason_for_call,\
                       LPVOID lpReserved)\
{return TRUE;}')

    cl_cmd += [dllmain_path]

    temp_path = dllmain_path.replace("dllmain.cc", "")
    cl_cmd += ["-Fo:" + temp_path]
    try:
        proc = subprocess.Popen(
            cl_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError("Can not find cl.exe,"
                           "please run this in Vistual Studio Command Prompt.")
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    link_cmd = ["lld-link"]
    link_cmd += ["-dll", "-FORCE:MULTIPLE"]

    for obj in objects:
        if obj.endswith(".cc"):
            (_, temp_file_name) = os.path.split(obj)
            (shot_name, _) = os.path.splitext(temp_file_name)
            link_cmd += [os.path.join(temp_path, shot_name + ".obj")]
        if obj.endswith(".o"):
            link_cmd += [obj]

    link_cmd += ["-EXPORT:__tvm_main__"]
    link_cmd += [temp_path + "dllmain.obj"]
    link_cmd += ["-out:" + output]

    try:
        proc = subprocess.Popen(
            link_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError("Can not find the LLVM linker for Windows (lld-link.exe)."
                           "Make sure it's installed"
                           " and the installation directory is in the %PATH% environment "
                           "variable. Prebuilt binaries can be found at: https://llvm.org/"
                           "For building the linker on your own see: https://lld.llvm.org/#build")
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)

        raise RuntimeError(msg)
