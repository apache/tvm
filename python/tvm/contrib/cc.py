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
"""Util to invoke C/C++ compilers in the system."""
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

    objects : List[str]
        List of object files.

    options : List[str]
        The list of additional options string.

    cc : Optional[str]
        The compiler command.
    """
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        _linux_compile(output, objects, options, cc)
    elif sys.platform == "win32":
        _windows_shared(output, objects, options)
    else:
        raise ValueError("Unsupported platform")

def get_target_by_dump_machine(compiler):
    """ Functor of get_target_triple that can get the target triple using compiler.

    Parameters
    ----------
    compiler : Optional[str]
        The compiler.

    Returns
    -------
    out: Callable
        A function that can get target triple according to dumpmachine option of compiler.
    """
    def get_target_triple():
        """ Get target triple according to dumpmachine option of compiler."""
        if compiler:
            cmd = [compiler, "-dumpmachine"]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                msg = "dumpmachine error:\n"
                msg += py_str(out)
                return None
            return py_str(out)
        return None

    return get_target_triple


# assign so as default output format
create_shared.output_format = "so" if sys.platform != "win32" else "dll"
create_shared.get_target_triple = get_target_by_dump_machine(
    "g++" if sys.platform == "darwin" or sys.platform.startswith("linux") else None)


def cross_compiler(compile_func,
                   options=None,
                   output_format=None,
                   get_target_triple=None,
                   add_files=None):
    """Create a cross compiler function by specializing compile_func with options.

    This function can be used to construct compile functions that
    can be passed to AutoTVM measure or export_library.


    Parameters
    ----------
    compile_func : Union[str, Callable[[str, str, Optional[str]], None]]
        Function that performs the actual compilation

    options : Optional[List[str]]
        List of additional optional string.

    output_format : Optional[str]
        Library output format.

    get_target_triple: Optional[Callable]
        Function that can target triple according to dumpmachine option of compiler.

    add_files: Optional[List[str]]
        List of paths to additional object, source, library files
        to pass as part of the compilation.

    Returns
    -------
    fcompile : Callable[[str, str, Optional[str]], None]
        A compilation function that can be passed to export_library.

    Examples
    --------
    .. code-block:: python

       from tvm.contrib import cc, ndk
       # export using arm gcc
       mod = build_runtime_module()
       mod.export_library(path_dso,
                          cc.cross_compiler("arm-linux-gnueabihf-gcc"))
       # specialize ndk compilation options.
       specialized_ndk = cc.cross_compiler(
           ndk.create_shared,
           ["--sysroot=/path/to/sysroot", "-shared", "-fPIC", "-lm"])
       mod.export_library(path_dso, specialized_ndk)
    """
    base_options = [] if options is None else options
    kwargs = {}
    add_files = [] if add_files is None else add_files

    # handle case where compile_func is the name of the cc
    if isinstance(compile_func, str):
        kwargs = {"cc" : compile_func}
        compile_func = create_shared


    def _fcompile(outputs, objects, options=None):
        all_options = base_options
        if options is not None:
            all_options += options
        compile_func(outputs, objects + add_files, options=all_options, **kwargs)

    if not output_format and hasattr(compile_func, "output_format"):
        output_format = compile_func.output_format
    output_format = output_format if output_format else "so"

    if not get_target_triple and hasattr(compile_func, "get_target_triple"):
        get_target_triple = compile_func.get_target_triple

    _fcompile.output_format = output_format
    _fcompile.get_target_triple = get_target_triple
    return _fcompile


def _linux_compile(output, objects, options, compile_cmd="g++"):
    cmd = [compile_cmd]
    if output.endswith(".so") or output.endswith(".dylib"):
        cmd += ["-shared", "-fPIC"]
        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]
    elif output.endswith(".obj"):
        cmd += ["-c"]
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
