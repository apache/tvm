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
import sys
import os
import subprocess

from .._ffi.base import py_str


def create_shared(output, objects, options=None, cc="g++"):
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
    if (
        sys.platform == "darwin"
        or sys.platform.startswith("linux")
        or sys.platform.startswith("freebsd")
    ):
        _linux_compile(output, objects, options, cc, compile_shared=True)
    elif sys.platform == "win32":
        _windows_compile(output, objects, options)
    else:
        raise ValueError("Unsupported platform")


def create_executable(output, objects, options=None, cc="g++"):
    """Create executable binary.

    Parameters
    ----------
    output : str
        The target executable.

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
        _windows_compile(output, objects, options)
    else:
        raise ValueError("Unsupported platform")


def get_target_by_dump_machine(compiler):
    """Functor of get_target_triple that can get the target triple using compiler.

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
        """Get target triple according to dumpmachine option of compiler."""
        if compiler:
            cmd = [compiler, "-dumpmachine"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    os.environ.get(
        "CXX", "g++" if sys.platform == "darwin" or sys.platform.startswith("linux") else None
    )
)


def cross_compiler(
    compile_func, options=None, output_format=None, get_target_triple=None, add_files=None
):
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
        kwargs = {"cc": compile_func}
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


def _linux_compile(output, objects, options, compile_cmd="g++", compile_shared=False):
    cmd = [compile_cmd]
    if compile_cmd != "nvcc":
        if compile_shared or output.endswith(".so") or output.endswith(".dylib"):
            cmd += ["-shared", "-fPIC"]
            if sys.platform == "darwin":
                cmd += ["-undefined", "dynamic_lookup"]
        elif output.endswith(".obj"):
            cmd += ["-c"]
    else:
        if compile_shared or output.endswith(".so") or output.endswith(".dylib"):
            cmd += ["--shared"]
    cmd += ["-o", output]
    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects
    if options:
        cmd += options
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        msg += "\nCommand line: " + " ".join(cmd)
        raise RuntimeError(msg)


def _windows_compile(output, objects, options):
    cmd = ["clang"]
    cmd += ["-O2"]

    if output.endswith(".so") or output.endswith(".dll"):
        cmd += ["-shared"]
    elif output.endswith(".obj"):
        cmd += ["-c"]

    if isinstance(objects, str):
        objects = [objects]
    cmd += ["-o", output]
    cmd += objects
    if options:
        cmd += options

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError(
            "Can not find the LLVM clang for Windows clang.exe)."
            "Make sure it's installed"
            " and the installation directory is in the %PATH% environment "
            "variable. Prebuilt binaries can be found at: https://llvm.org/"
        )
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += " ".join(cmd) + "\n"
        msg += py_str(out)

        raise RuntimeError(msg)
